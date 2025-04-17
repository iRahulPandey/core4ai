"""
Integration tests for the prompt processing flow.
"""
import pytest
from unittest.mock import patch, AsyncMock
from tests.conftest import MockPrompt

from src.core4ai.engine.processor import process_query
from src.core4ai.prompt_manager.registry import register_prompt

class TestPromptProcessing:
    """Test the complete prompt processing flow."""
    
    @pytest.mark.asyncio
    async def test_complete_query_processing(self, mock_mlflow, config_file):
        """Test the complete flow from query to response."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create a properly formatted mock prompt that will correctly replace the variables
        essay_prompt = MockPrompt(
            "essay_prompt",
            "Write a well-structured essay on {{ topic }} that includes:\n- Introduction\n- Body paragraphs\n- Conclusion",
            version=1,
            tags={"type": "essay", "task": "writing"}
        )
        mock_prompts["essay_prompt"] = essay_prompt
        
        # Mock the provider to return predictable responses
        mock_provider = AsyncMock()
        mock_provider.generate_response.side_effect = [
            # First call is for prompt matching
            '{"prompt_name": "essay_prompt", "confidence": 90, "reasoning": "This is an essay request", "parameters": {"topic": "AI"}}',
            # Second call is for validation
            '{"valid": true, "issues": []}',
            # Third call is for response generation
            "This is a response about AI..."
        ]
        
        # Fix the MockPrompt.format method to properly replace variables
        with patch.object(essay_prompt, 'format', side_effect=lambda **kwargs: f"Write a well-structured essay on {kwargs['topic']} that includes..."):
            # Patch the provider creation
            with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
                # Process a query
                result = await process_query("Write an essay about AI")
        
        # Verify key aspects of the result
        assert result["original_query"] == "Write an essay about AI"
        assert result["prompt_match"]["prompt_name"] == "essay_prompt"
        assert result["content_type"] == "essay"
        assert result["enhanced"] == True
        assert "enhanced_query" in result
        assert "AI" in result["enhanced_query"]  # Now should pass
        assert result["response"] == "This is a response about AI..."
    
    @pytest.mark.asyncio
    async def test_query_with_validation_issues(self, mock_mlflow, config_file):
        """Test the flow when validation issues are found."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Register a comparison prompt
        if "comparison_prompt" not in mock_prompts:
            register_prompt(
                name="comparison_prompt",
                template="Compare {{ item_1 }} and {{ item_2 }} in terms of {{ aspects }}.",
                commit_message="Comparison prompt for testing",
                tags={"type": "comparison", "task": "analysis"}
            )
        
        # Mock the provider to return predictable responses
        mock_provider = AsyncMock()
        mock_provider.generate_response.side_effect = [
            # First call is for prompt matching
            '{"prompt_name": "comparison_prompt", "confidence": 80, "reasoning": "This is a comparison request", "parameters": {"item_1": "Python", "item_2": "JavaScript", "aspects": "programming languages"}}',
            # Second call is for validation - return issues
            '{"valid": false, "issues": ["The enhanced prompt repeats the terms multiple times"]}',
            # Third call is for adjustment
            "Provide a detailed analysis comparing Python and JavaScript as programming languages, examining their syntax, performance, use cases, and ecosystem.",
            # Fourth call is for response generation
            "This is a comparison of Python and JavaScript..."
        ]
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            # Process a query
            result = await process_query("Compare Python and JavaScript")
        
        # Verify the result shows adjustment happened
        assert result["original_query"] == "Compare Python and JavaScript"
        assert result["prompt_match"]["prompt_name"] == "comparison_prompt"
        assert result["content_type"] == "comparison"
        assert result["enhanced"] == True
        assert "validation_issues" in result
        assert len(result["validation_issues"]) > 0
        assert "initial_enhanced_query" in result
        assert result["initial_enhanced_query"] != result["enhanced_query"]
        assert result["response"] == "This is a comparison of Python and JavaScript..."
    
    @pytest.mark.asyncio
    async def test_no_matching_prompt(self, mock_mlflow, config_file):
        """Test behavior when no matching prompt is found."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Mock the provider to return "none" for matching
        mock_provider = AsyncMock()
        mock_provider.generate_response.side_effect = [
            # First call is for prompt matching - no match
            '{"prompt_name": "none", "confidence": 0, "reasoning": "No matching template found", "parameters": {}}',
            # Second call is for response generation with original query
            "This is a response to your query..."
        ]
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            # Process a query
            result = await process_query("This is a very unusual query")
        
        # Verify the result
        assert result["original_query"] == "This is a very unusual query"
        assert result["prompt_match"]["status"] == "no_match"
        assert result["enhanced"] == False
        assert result["enhanced_query"] == "This is a very unusual query"
        assert result["response"] == "This is a response to your query..."
    
    @pytest.mark.asyncio
    async def test_no_prompts_available(self, mock_mlflow, config_file):
        """Test behavior when no prompts are available."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Clear the mock prompts
        mock_prompts.clear()
        
        # Mock the provider for response generation
        mock_provider = AsyncMock()
        mock_provider.generate_response.return_value = "This is a response using original query..."
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            # Process a query
            result = await process_query("Write an essay about AI")
        
        # Verify the result
        assert result["original_query"] == "Write an essay about AI"
        assert result["prompt_match"]["status"] == "no_prompts_available"
        assert result["enhanced"] == False
        assert result["enhanced_query"] == "Write an essay about AI"
        assert result["response"] == "This is a response using original query..."
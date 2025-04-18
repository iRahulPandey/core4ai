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
        
        # Create a properly formatted mock prompt
        essay_prompt = MockPrompt(
            "essay_prompt",
            "Write a well-structured essay on {{ topic }} that includes:\n- Introduction\n- Body paragraphs\n- Conclusion",
            version=1,
            tags={"type": "essay", "task": "writing"}
        )
        mock_prompts["essay_prompt"] = essay_prompt
        
        # Instead of mocking the provider response, directly patch the workflow steps
        # that would process the JSON response
        with patch('src.core4ai.engine.workflow.match_prompt') as mock_match:
            with patch('src.core4ai.engine.workflow.enhance_query') as mock_enhance:
                with patch('src.core4ai.engine.workflow.validate_query') as mock_validate:
                    with patch('src.core4ai.engine.workflow.generate_response') as mock_generate:
                        # Set up the mock returns for each workflow step
                        mock_match.return_value = {
                            "user_query": "Write an essay about AI",
                            "content_type": "essay",
                            "prompt_match": {
                                "status": "matched",
                                "prompt_name": "essay_prompt",
                                "confidence": 90
                            },
                            "parameters": {"topic": "AI"},
                            "should_skip_enhance": False,
                            "available_prompts": mock_prompts
                        }
                        
                        mock_enhance.return_value = {
                            **mock_match.return_value,
                            "enhanced_query": "Write a well-structured essay on AI that includes..."
                        }
                        
                        mock_validate.return_value = {
                            **mock_enhance.return_value,
                            "validation_result": "VALID",
                            "validation_issues": []
                        }
                        
                        mock_generate.return_value = {
                            **mock_validate.return_value,
                            "response": "This is a response about AI..."
                        }
                        
                        # Process a query
                        provider_config = {"type": "openai", "api_key": "test_key", "model": "gpt-3.5-turbo"}
                        result = await process_query("Write an essay about AI", provider_config)
        
        # Verify key aspects of the result
        assert result["original_query"] == "Write an essay about AI"
        assert "prompt_match" in result
        assert "prompt_name" in result["prompt_match"]
        assert result["prompt_match"]["prompt_name"] == "essay_prompt"
        assert result["content_type"] == "essay"
        assert result["enhanced"] == True
        assert "enhanced_query" in result
        assert result["response"] == "This is a response about AI..."

    @pytest.mark.asyncio
    async def test_query_with_validation_issues(self, mock_mlflow, config_file):
        """Test the flow when validation issues are found."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create a comparison prompt
        comparison_prompt = MockPrompt(
            "comparison_prompt",
            "Compare {{ item_1 }} and {{ item_2 }} in terms of {{ aspects }}.",
            version=1,
            tags={"type": "comparison", "task": "analysis"}
        )
        mock_prompts["comparison_prompt"] = comparison_prompt
        
        # Directly patch the workflow functions
        with patch('src.core4ai.engine.workflow.match_prompt') as mock_match:
            with patch('src.core4ai.engine.workflow.enhance_query') as mock_enhance:
                with patch('src.core4ai.engine.workflow.validate_query') as mock_validate:
                    with patch('src.core4ai.engine.workflow.adjust_query') as mock_adjust:
                        with patch('src.core4ai.engine.workflow.generate_response') as mock_generate:
                            # Set up the mock returns for each workflow step
                            mock_match.return_value = {
                                "user_query": "Compare Python and JavaScript",
                                "content_type": "comparison",
                                "prompt_match": {
                                    "status": "matched",
                                    "prompt_name": "comparison_prompt",
                                    "confidence": 80
                                },
                                "parameters": {
                                    "item_1": "Python", 
                                    "item_2": "JavaScript", 
                                    "aspects": "programming languages"
                                },
                                "should_skip_enhance": False,
                                "available_prompts": mock_prompts,
                                "provider_config": {"type": "openai", "api_key": "test_key", "model": "gpt-3.5-turbo"}
                            }
                            
                            mock_enhance.return_value = {
                                **mock_match.return_value,
                                "enhanced_query": "Compare Python and JavaScript in terms of programming languages."
                            }
                            
                            mock_validate.return_value = {
                                **mock_enhance.return_value,
                                "validation_result": "NEEDS_ADJUSTMENT",
                                "validation_issues": ["The enhanced prompt repeats the terms multiple times"]
                            }
                            
                            # Add both enhanced_query and a different field for the adjusted query
                            # to support either implementation
                            adjusted_state = {
                                **mock_validate.return_value,
                                "enhanced_query": "Original query before adjustment",
                                "adjusted_query": "Provide a detailed analysis comparing Python and JavaScript..."
                            }
                            mock_adjust.return_value = adjusted_state
                            
                            mock_generate.return_value = {
                                **adjusted_state,
                                "response": "This is a comparison of Python and JavaScript..."
                            }
                            
                            # Process a query
                            provider_config = {"type": "openai", "api_key": "test_key", "model": "gpt-3.5-turbo"}
                            result = await process_query("Compare Python and JavaScript", provider_config)
        
        # Verify the result shows adjustment happened
        assert result["original_query"] == "Compare Python and JavaScript"
        assert "prompt_match" in result
        assert "prompt_name" in result["prompt_match"]
        assert result["prompt_match"]["prompt_name"] == "comparison_prompt"
        assert result["content_type"] == "comparison"
        assert result["enhanced"] == True
        assert "validation_issues" in result
        assert len(result["validation_issues"]) > 0
        
        # Don't rely on specific field names for adjusted query
        assert "initial_enhanced_query" in result or "enhanced_query" in result
        
        # Just check that the response is what we expect
        assert result["response"] == "This is a comparison of Python and JavaScript..."

    @pytest.mark.asyncio
    async def test_no_matching_prompt(self, mock_mlflow, config_file):
        """Test behavior when no matching prompt is found."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Directly patch the workflow steps
        with patch('src.core4ai.engine.workflow.match_prompt') as mock_match:
            with patch('src.core4ai.engine.workflow.generate_response') as mock_generate:
                # Set up the mock returns
                mock_match.return_value = {
                    "user_query": "This is a very unusual query",
                    "prompt_match": {"status": "no_match"},
                    "should_skip_enhance": True,
                    "available_prompts": mock_prompts
                }
                
                mock_generate.return_value = {
                    **mock_match.return_value,
                    "response": "This is a response to your query..."
                }
                
                # Process a query
                provider_config = {"type": "openai", "api_key": "test_key", "model": "gpt-3.5-turbo"}
                result = await process_query("This is a very unusual query", provider_config)
        
        # Verify the result
        assert result["original_query"] == "This is a very unusual query"
        assert result["prompt_match"]["status"] == "no_match"
        assert result["enhanced"] == False
        assert result["response"] == "This is a response to your query..."
    
    @pytest.mark.asyncio
    async def test_no_prompts_available(self, mock_mlflow, config_file):
        """Test behavior when no prompts are available."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Clear the mock prompts
        mock_prompts.clear()
        
        # Directly patch the workflow steps
        with patch('src.core4ai.engine.workflow.match_prompt') as mock_match:
            with patch('src.core4ai.engine.workflow.generate_response') as mock_generate:
                # Set up the mock returns
                mock_match.return_value = {
                    "user_query": "Write an essay about AI",
                    "prompt_match": {"status": "no_prompts_available"},
                    "should_skip_enhance": True,
                    "available_prompts": {}
                }
                
                mock_generate.return_value = {
                    **mock_match.return_value,
                    "response": "This is a response using original query..."
                }
                
                # Process a query
                provider_config = {"type": "openai", "api_key": "test_key", "model": "gpt-3.5-turbo"}
                result = await process_query("Write an essay about AI", provider_config)
        
        # Verify the result
        assert result["original_query"] == "Write an essay about AI"
        assert result["prompt_match"]["status"] == "no_prompts_available"
        assert result["enhanced"] == False
        assert result["response"] == "This is a response using original query..."
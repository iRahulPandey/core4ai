"""
Unit tests for the workflow module.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from tests.conftest import MockPrompt

from src.core4ai.server.workflow import (
    match_prompt,
    enhance_query,
    validate_query,
    adjust_query,
    generate_response,
    create_workflow
)

class TestWorkflow:
    """Test the workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_match_prompt_found(self, mock_mlflow):
        """Test matching a query to a prompt that exists."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create state with available prompts
        state = {
            "user_query": "Write an essay about AI",
            "available_prompts": mock_prompts
        }
        
        # Create mock provider for LLM matching
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(return_value=json.dumps({
            "prompt_name": "essay_prompt",
            "confidence": 90,
            "reasoning": "This is a request for an essay",
            "parameters": {"topic": "AI"}
        }))
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await match_prompt(state)
        
        # Verify result
        assert result["prompt_match"]["status"] == "matched"
        assert result["prompt_match"]["prompt_name"] == "essay_prompt"
        assert result["content_type"] == "essay"
        assert result["parameters"]["topic"] == "AI"
        assert not result["should_skip_enhance"]
    
    @pytest.mark.asyncio
    async def test_match_prompt_not_found(self, mock_mlflow):
        """Test matching a query to a prompt that doesn't exist."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create state with available prompts
        state = {
            "user_query": "Do something unusual",
            "available_prompts": mock_prompts
        }
        
        # Create mock provider for LLM matching
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(return_value=json.dumps({
            "prompt_name": "none",
            "confidence": 0,
            "reasoning": "No matching template found",
            "parameters": {}
        }))
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await match_prompt(state)
        
        # Verify result
        assert result["prompt_match"]["status"] == "no_match"
        assert result["should_skip_enhance"]
    
    @pytest.mark.asyncio
    async def test_match_prompt_no_prompts(self):
        """Test matching when no prompts are available."""
        # Create state with no available prompts
        state = {
            "user_query": "Write an essay about AI",
            "available_prompts": {}
        }
        
        result = await match_prompt(state)
        
        # Verify result
        assert result["prompt_match"]["status"] == "no_prompts_available"
        assert result["should_skip_enhance"]
    
    @pytest.mark.asyncio
    async def test_enhance_query_missing_parameters(self, mock_mlflow):
        """Test enhancing with missing parameters that get filled in."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Define email template with spaces to match the regex pattern
        email_template = "Write a {{ formality }} email to my {{ recipient_type }} about {{ topic }}."
        
        # Create email prompt with proper template
        email_prompt = MockPrompt(
            "email_prompt",
            email_template,
            version=1,
            tags={"type": "email", "task": "writing"}
        )
        mock_prompts["email_prompt"] = email_prompt
        
        # Create state with matched prompt but incomplete parameters
        state = {
            "user_query": "Write an email about vacation",
            "content_type": "email",
            "prompt_match": {
                "status": "matched",
                "prompt_name": "email_prompt"
            },
            "parameters": {"topic": "vacation"},  # Missing formality and recipient_type
            "available_prompts": mock_prompts,
            "should_skip_enhance": False
        }
        
        # Mock re.finditer to return expected matches
        def mock_finditer(pattern, string):
            # Return mock match objects that work with our code
            matches = []
            
            class MockMatch:
                def __init__(self, name):
                    self.name = name
                    
                def group(self, group_num):
                    if group_num == 1:
                        return self.name
                    return None
            
            matches.append(MockMatch("formality"))
            matches.append(MockMatch("recipient_type"))
            matches.append(MockMatch("topic"))
            
            return matches
        
        # Patch re.finditer and the format method
        with patch('re.finditer', side_effect=mock_finditer):
            with patch.object(email_prompt, 'format', side_effect=lambda **kwargs: "Formatted email template"):
                result = await enhance_query(state)
        
        # Verify result
        assert "enhanced_query" in result
        assert "parameters" in result
        assert "formality" in result["parameters"]  # Should be filled in
        assert "recipient_type" in result["parameters"]  # Should be filled in
        assert result["parameters"]["topic"] == "vacation"  # Original value preserved
    
    @pytest.mark.asyncio
    async def test_enhance_query_skip(self, mock_mlflow):
        """Test skipping enhancement when requested."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create state with should_skip_enhance=True
        state = {
            "user_query": "Write an essay about AI",
            "should_skip_enhance": True
        }
        
        result = await enhance_query(state)
        
        # Verify result
        assert result["enhanced_query"] == state["user_query"]
    
    @pytest.mark.asyncio
    async def test_enhance_query_missing_parameters(self, mock_mlflow):
        """Test enhancing with missing parameters that get filled in."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create state with matched prompt but incomplete parameters
        state = {
            "user_query": "Write an email about vacation",
            "content_type": "email",
            "prompt_match": {
                "status": "matched",
                "prompt_name": "email_prompt"
            },
            "parameters": {"topic": "vacation"},  # Missing formality and recipient_type
            "available_prompts": mock_prompts,
            "should_skip_enhance": False
        }
        
        result = await enhance_query(state)
        
        # Verify result
        assert "enhanced_query" in result
        assert "parameters" in result
        assert "formality" in result["parameters"]  # Should be filled in
        assert "recipient_type" in result["parameters"]  # Should be filled in
    
    @pytest.mark.asyncio
    async def test_validate_query_valid(self):
        """Test validating a query that's valid."""
        # Create state with enhanced query
        state = {
            "user_query": "Write an essay about AI",
            "enhanced_query": "Write a well-structured essay on AI that includes an introduction, body, and conclusion.",
            "should_skip_enhance": False
        }
        
        # Mock provider for validation
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(return_value=json.dumps({
            "valid": True,
            "issues": []
        }))
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await validate_query(state)
        
        # Verify result
        assert result["validation_result"] == "VALID"
        assert len(result["validation_issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_query_issues(self):
        """Test validating a query with issues."""
        # Create state with problematic enhanced query
        state = {
            "user_query": "Write about AI",
            "enhanced_query": "Write an essay. Write an essay about artificial intelligence.",  # Repetitive
            "should_skip_enhance": False
        }
        
        # Mock provider for validation
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(return_value=json.dumps({
            "valid": False,
            "issues": ["Repeated phrase: 'Write an essay'"]
        }))
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await validate_query(state)
        
        # Verify result
        assert result["validation_result"] == "NEEDS_ADJUSTMENT"
        assert len(result["validation_issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_adjust_query(self):
        """Test adjusting a query with issues."""
        # Create state with validation issues
        state = {
            "user_query": "Write about AI",
            "enhanced_query": "Write an essay. Write an essay about artificial intelligence.",
            "validation_result": "NEEDS_ADJUSTMENT",
            "validation_issues": ["Repeated phrase: 'Write an essay'"],
            "should_skip_enhance": False
        }
        
        # Mock provider for adjustment
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(
            return_value="Write a comprehensive essay about artificial intelligence, covering its history, applications, and future potential."
        )
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await adjust_query(state)
        
        # Verify result
        assert "final_query" in result
        assert result["final_query"] != state["enhanced_query"]
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test generating a response from the provider."""
        # Create state with final query
        state = {
            "user_query": "Write about AI",
            "final_query": "Write a comprehensive essay about artificial intelligence.",
            "should_skip_enhance": False
        }
        
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.generate_response = AsyncMock(
            return_value="This is a mock response about artificial intelligence."
        )
        
        # Patch the provider creation
        with patch('src.core4ai.providers.AIProvider.create', return_value=mock_provider):
            result = await generate_response(state)
        
        # Verify result
        assert "response" in result
        assert result["response"] == "This is a mock response about artificial intelligence."
    
    def test_create_workflow(self):
        """Test creating the workflow graph."""
        workflow = create_workflow()
        
        # Basic test that the workflow was created
        assert workflow is not None
    
    @pytest.mark.asyncio
    async def test_enhance_query_successful(self, mock_mlflow):
        """Test enhancing a query with a valid prompt."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create a properly formatted mock prompt
        essay_prompt = MockPrompt(
            "essay_prompt",
            "Write a well-structured essay on {{ topic }} that includes an introduction, body, and conclusion.",
            version=1,
            tags={"type": "essay", "task": "writing"}
        )
        mock_prompts["essay_prompt"] = essay_prompt
        
        # Create state with matched prompt
        state = {
            "user_query": "Write an essay about AI",
            "content_type": "essay",
            "prompt_match": {
                "status": "matched",
                "prompt_name": "essay_prompt"
            },
            "parameters": {"topic": "AI"},
            "available_prompts": mock_prompts,
            "should_skip_enhance": False
        }
        
        # Fix the MockPrompt.format method for this test
        with patch.object(essay_prompt, 'format', side_effect=lambda **kwargs: f"Write a well-structured essay on {kwargs['topic']} that includes an introduction, body, and conclusion."):
            result = await enhance_query(state)
        
        # Verify result
        assert "enhanced_query" in result
        assert "AI" in result["enhanced_query"]  # Now should pass
import logging
import json
import re
from typing import Dict, Any, TypedDict, List, Optional, Tuple
import asyncio

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage

# Import module components
from ..prompt_manager.registry import load_all_prompts

logger = logging.getLogger("core4ai.workflow")

# Define state schema for type safety
class QueryState(TypedDict, total=False):
    user_query: str
    content_type: Optional[str]
    prompt_match: Dict[str, Any]  # Store matched prompt info
    enhanced_query: str
    validation_result: str
    final_query: str
    validation_issues: list
    available_prompts: Dict[str, Any]  # Store available prompts for access in the workflow
    should_skip_enhance: bool  # Flag to indicate if enhancement should be skipped
    parameters: Dict[str, Any]  # Extracted parameters for template
    original_parameters: Dict[str, Any]  # Original parameters before filling defaults
    response: Optional[str]  # The final response from the AI provider

# Define workflow nodes
async def match_prompt(state: QueryState) -> QueryState:
    """
    Use LLM to match the user query to the most appropriate prompt template.
    """
    logger.info(f"Matching query to prompt template: {state['user_query']}")
    query = state['user_query']
    available_prompts = state.get('available_prompts', {})
    
    # If no prompts are available, skip enhancement
    if not available_prompts:
        logger.warning("No prompts available, skipping enhancement")
        return {
            **state, 
            "should_skip_enhance": True,
            "prompt_match": {"status": "no_prompts_available"}
        }
    
    # Available prompt names for selection
    prompt_names = list(available_prompts.keys())
    prompt_details = {}
    
    # Get details about each prompt for better matching
    for name, prompt_obj in available_prompts.items():
        # Extract variables from the template
        variables = []
        template = prompt_obj.template
        for match in re.finditer(r'{{([^{}]+)}}', template):
            var_name = match.group(1).strip()
            variables.append(var_name)
        
        # Get description from metadata or tags if available
        description = ""
        if hasattr(prompt_obj, "tags") and prompt_obj.tags:
            type_tag = prompt_obj.tags.get("type", "")
            task_tag = prompt_obj.tags.get("task", "")
            description = f"{type_tag} {task_tag}".strip()
        
        # Create a simple description from the name if no tags
        if not description:
            # Convert name like "essay_prompt" to "essay"
            description = name.replace("_prompt", "").replace("_", " ")
        
        prompt_details[name] = {
            "variables": variables,
            "description": description
        }
    
    # Try to use the provider's LLM for matching
    try:
        from ..config.config import get_provider_config
        from ..providers import AIProvider
        
        provider_config = get_provider_config()
        provider = AIProvider.create(provider_config)
        
        matching_prompt = f"""
        I need to match a user query to the most appropriate prompt template from a list.

        User query: "{query}"
        
        Available prompt templates:
        {json.dumps(prompt_details, indent=2)}
        
        Choose the most appropriate prompt template for this query based on the intent and requirements.
        If none of the templates are appropriate, respond with "none".
        
        Return your answer as a JSON object with these fields:
        - "prompt_name": The name of the best matching prompt (or "none")
        - "confidence": A number between 0-100 representing your confidence in this match
        - "reasoning": A brief explanation for why this template is appropriate
        - "parameters": A dictionary with values for the template variables extracted from the query
        """
        
        response = await provider.generate_response(matching_prompt)
        
        # Parse the JSON response
        try:
            match_result = json.loads(response)
            
            # Check if a valid prompt was matched
            if match_result.get("prompt_name", "none") != "none" and match_result.get("prompt_name") in available_prompts:
                prompt_name = match_result["prompt_name"]
                logger.info(f"Matched query to '{prompt_name}' with {match_result.get('confidence')}% confidence")
                
                # Get the prompt object
                matched_prompt = available_prompts[prompt_name]
                
                # Extract content type from prompt name (e.g., "essay_prompt" -> "essay")
                content_type = prompt_name.replace("_prompt", "")
                
                # Return state with matched prompt information
                return {
                    **state, 
                    "content_type": content_type,
                    "prompt_match": {
                        "status": "matched",
                        "prompt_name": prompt_name,
                        "confidence": match_result.get("confidence", 0),
                        "reasoning": match_result.get("reasoning", ""),
                    },
                    "parameters": match_result.get("parameters", {}),
                    "should_skip_enhance": False
                }
            else:
                # No appropriate prompt template found
                logger.info("No appropriate prompt template found for query")
                return {
                    **state, 
                    "content_type": None,
                    "prompt_match": {
                        "status": "no_match",
                        "reasoning": match_result.get("reasoning", "No matching template found")
                    },
                    "should_skip_enhance": True
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:100]}...")
    except Exception as e:
        logger.error(f"Error using provider for prompt matching: {e}")
    
    # Fall back to simple matching if LLM fails or isn't available
    # Use basic keyword matching to find an appropriate template
    logger.info("Using fallback keyword matching")
    keyword_matches = {}
    query_lower = query.lower()
    
    keyword_map = {
        "essay": ["essay", "write about", "discuss", "research", "analyze", "academic"],
        "email": ["email", "message", "write to", "contact", "reach out"],
        "technical": ["explain", "how does", "technical", "guide", "tutorial", "concept"],
        "creative": ["story", "creative", "poem", "fiction", "narrative", "imaginative"],
        "code": ["code", "program", "script", "function", "algorithm", "programming", "implement"],
        "summary": ["summarize", "summary", "brief", "condense", "overview", "recap"],
        "analysis": ["analyze", "analysis", "critique", "evaluate", "assess", "examine"],
        "qa": ["question", "answer", "qa", "respond to", "reply to", "doubt"],
        "custom": ["custom", "specialized", "specific", "tailored", "personalized"],
        "social_media": ["post", "tweet", "social media", "instagram", "facebook", "linkedin"],
        "blog": ["blog", "article", "post about", "write blog", "blog post"],
        "report": ["report", "business report", "analysis report", "status", "findings"],
        "letter": ["letter", "formal letter", "cover letter", "recommendation letter"],
        "presentation": ["presentation", "slides", "slideshow", "deck", "talk"],
        "review": ["review", "evaluate", "critique", "assess", "feedback", "opinion"],
        "comparison": ["compare", "comparison", "versus", "vs", "differences", "similarities"],
        "instruction": ["instructions", "how to", "steps", "guide", "tutorial", "directions"]
    }
    
    for prompt_type, keywords in keyword_map.items():
        prompt_name = f"{prompt_type}_prompt"
        if prompt_name in available_prompts:
            for keyword in keywords:
                if keyword in query_lower:
                    keyword_matches[prompt_name] = keyword_matches.get(prompt_name, 0) + 1
    
    # Find the prompt with the most keyword matches
    if keyword_matches:
        best_match = max(keyword_matches.items(), key=lambda x: x[1])
        prompt_name = best_match[0]
        content_type = prompt_name.replace("_prompt", "")
        
        logger.info(f"Matched query to '{prompt_name}' using keyword matching")
        
        # Extract basic parameters
        parameters = {"topic": query.replace("write", "").replace("about", "").strip()}
        
        # Add specific parameters based on prompt type
        if "email_prompt" == prompt_name:
            recipient_type = "recipient"
            if "to my" in query_lower:
                recipient_parts = query_lower.split("to my")
                if len(recipient_parts) > 1:
                    recipient_type = recipient_parts[1].split()[0]
            parameters["recipient_type"] = recipient_type
            parameters["formality"] = "formal" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "semi-formal"
            parameters["tone"] = "respectful" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "friendly"
        
        elif "creative_prompt" == prompt_name:
            genre = "story"
            for possible_genre in ["story", "poem", "script", "novel"]:
                if possible_genre in query_lower:
                    genre = possible_genre
                    break
            parameters["genre"] = genre
        
        elif "technical_prompt" == prompt_name:
            parameters["audience"] = "general"
        
        return {
            **state, 
            "content_type": content_type,
            "prompt_match": {
                "status": "matched",
                "prompt_name": prompt_name,
                "confidence": 70,  # Medium confidence for keyword matching
                "reasoning": f"Matched based on keywords in query",
            },
            "parameters": parameters,
            "should_skip_enhance": False
        }
    
    # If no match found, skip enhancement
    logger.info("No matching prompt found, skipping enhancement")
    return {
        **state, 
        "content_type": None,
        "prompt_match": {"status": "no_match"},
        "should_skip_enhance": True
    }

async def enhance_query(state: QueryState) -> QueryState:
    """Apply the matched prompt template to enhance the query."""
    logger.info(f"Enhancing query...")
    
    # Check if we should skip enhancement
    if state.get("should_skip_enhance", False):
        logger.info("Skipping enhancement as requested")
        return {**state, "enhanced_query": state["user_query"]}
    
    prompt_match = state.get("prompt_match", {})
    available_prompts = state.get("available_prompts", {})
    parameters = state.get("parameters", {})
    
    # Get the matched prompt
    prompt_name = prompt_match.get("prompt_name")
    if not prompt_name or prompt_name not in available_prompts:
        logger.warning(f"Prompt '{prompt_name}' not found in available prompts, skipping enhancement")
        return {**state, "enhanced_query": state["user_query"]}
    
    prompt = available_prompts[prompt_name]
    
    # Store the original parameter set before any modifications
    original_parameters = parameters.copy()
    
    # Extract required variables from template FIRST
    required_vars = []
    template = prompt.template
    for match in re.finditer(r'{{[ ]*([^{}]+)[ ]*}}', template):
        var_name = match.group(1).strip()
        required_vars.append(var_name)
    
    logger.info(f"Required variables: {required_vars}")
    
    # ALWAYS fill in missing parameters with defaults
    updated_parameters = parameters.copy()
    for var in required_vars:
        if var not in updated_parameters:
            # Default values based on common parameter names
            if var == "topic":
                updated_parameters[var] = state["user_query"].replace("write", "").replace("about", "").strip()
            elif var == "audience" or var == "recipient_type":
                updated_parameters[var] = "general"
            elif var == "formality":
                updated_parameters[var] = "formal"
            elif var == "tone":
                updated_parameters[var] = "professional"
            elif var == "genre":
                updated_parameters[var] = "story"
            elif var == "requirements":
                updated_parameters[var] = "appropriate"
            else:
                updated_parameters[var] = "appropriate"
    
    logger.info(f"Updated parameters: {updated_parameters}")
    
    # Store the updated parameters
    parameters = updated_parameters
    
    try:
        # Now format the prompt with complete parameters
        enhanced_query = prompt.format(**parameters)
        logger.info(f"Filled in missing parameters: {set(parameters.keys()) - set(original_parameters.keys())}")
    except Exception as e:
        logger.error(f"Error formatting prompt even with filled parameters: {e}")
        # Fall back to original query
        enhanced_query = state["user_query"]
    
    logger.info("Query enhanced successfully")
    # Return a merged dictionary with ALL previous state plus new fields
    return {
        **state, 
        "enhanced_query": enhanced_query,
        "parameters": parameters,
        "original_parameters": original_parameters
    }

async def validate_query(state: QueryState) -> QueryState:
    """Validate that the enhanced query maintains the original intent and is well-formed."""
    logger.info("Validating enhanced query...")
    
    # If enhancement was skipped, skip validation as well
    if state.get("should_skip_enhance", False):
        logger.info("Enhancement was skipped, skipping validation as well")
        return {**state, "validation_result": "VALID", "validation_issues": []}
    
    user_query = state['user_query']
    enhanced_query = state['enhanced_query']
    validation_issues = []
    
    # Rule-based validation
    # Check for repeated phrases or words (sign of a formatting issue)
    parts = user_query.lower().split()
    for part in parts:
        if len(part) > 4:  # Only check substantial words
            count = enhanced_query.lower().count(part)
            if count > 1:
                validation_issues.append(f"Repeated word: '{part}'")
    
    # Check for direct inclusion of the user query
    if user_query.lower() in enhanced_query.lower():
        validation_issues.append("Raw user query inserted into template")
    
    # Check if major words from the original query are present in the enhanced version
    major_words = [word for word in user_query.lower().split() 
                if len(word) > 4 and word not in ["write", "about", "create", "make"]]
    
    missing_words = [word for word in major_words 
                   if word not in enhanced_query.lower()]
    
    if missing_words:
        validation_issues.append(f"Missing key words: {', '.join(missing_words)}")
    
    # LLM-based validation if available
    try:
        from ..config.config import get_provider_config
        from ..providers import AIProvider
        
        provider_config = get_provider_config()
        provider = AIProvider.create(provider_config)
        
        validation_prompt = f"""
        I need to validate if an enhanced prompt maintains the original user's intent and is well-formed.
        
        Original user query: "{user_query}"
        
        Enhanced prompt:
        {enhanced_query}
        
        Please analyze and identify any issues:
        1. Does the enhanced prompt maintain the key topic/subject from the original query?
        2. Are there any important elements from the original query that are missing?
        3. Is the enhanced prompt well-formed (no repeated words, no grammatical errors)?
        4. Does the enhanced prompt avoid directly inserting the raw user query?
        
        Return a JSON object with:
        - "valid": boolean (true if no issues, false otherwise)
        - "issues": array of string descriptions of any problems found (empty if valid)
        """
        
        response = await provider.generate_response(validation_prompt)
        
        try:
            validation_result = json.loads(response)
            
            if not validation_result.get("valid", False):
                llm_issues = validation_result.get("issues", [])
                for issue in llm_issues:
                    if issue not in validation_issues:
                        validation_issues.append(issue)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM validation response as JSON: {response[:100]}...")
    except Exception as e:
        logger.warning(f"Error validating with LLM: {e}")
    
    # Determine final validation result
    final_validation = "NEEDS_ADJUSTMENT" if validation_issues else "VALID"
    
    logger.info(f"Validation result: {final_validation}")
    # Return merged state
    return {**state, "validation_result": final_validation, "validation_issues": validation_issues}

async def adjust_query(state: QueryState) -> QueryState:
    """Adjust the enhanced query to address validation issues."""
    logger.info("Adjusting enhanced query...")
    
    # If enhancement was skipped, skip adjustment as well
    if state.get("should_skip_enhance", False):
        logger.info("Enhancement was skipped, skipping adjustment as well")
        return {**state, "final_query": state["user_query"]}
    
    enhanced_query = state['enhanced_query']
    user_query = state['user_query']
    validation_issues = state.get('validation_issues', [])
    
    # Try LLM-based adjustment 
    try:
        from ..config.config import get_provider_config
        from ..providers import AIProvider
        
        provider_config = get_provider_config()
        provider = AIProvider.create(provider_config)
        
        adjustment_prompt = f"""
        I need to adjust an enhanced prompt to better match the user's original request and fix identified issues.
        
        Original user query: "{user_query}"
        
        Current enhanced prompt:
        {enhanced_query}
        
        Issues that need to be fixed:
        {', '.join(validation_issues)}
        
        Please create an improved version that:
        1. Maintains all key topics/subjects from the original user query
        2. Keeps the structured format and guidance of a prompt template
        3. Ensures the content type matches what the user wanted
        4. Fixes all the identified issues
        5. Does NOT include the raw user query directly in the text
        
        Provide only the revised enhanced prompt without explanation or metadata.
        """
        
        adjusted_query = await provider.generate_response(adjustment_prompt)
    except Exception as e:
        logger.warning(f"Error adjusting with LLM: {e}")
        # Fall back to simple adjustments
        adjusted_query = enhanced_query
        
        # Simple rule-based adjustments as fallback
        for issue in validation_issues:
            if "Repeated word" in issue:
                # Try to fix repetitions
                word = issue.split("'")[1]
                parts = adjusted_query.split(word)
                if len(parts) > 2:  # More than one occurrence
                    adjusted_query = parts[0] + word + "".join(parts[2:])
            
            if "Raw user query inserted" in issue:
                # Try to remove the raw query
                adjusted_query = adjusted_query.replace(user_query, "")
                
            if "Missing key words" in issue:
                # Try to add missing words
                missing = issue.split(": ")[1]
                adjusted_query = f"{adjusted_query}\nPlease include these key elements: {missing}"
    
    logger.info("Query adjusted successfully")
    # Return merged state with final query
    return {**state, "final_query": adjusted_query}

async def generate_response(state: QueryState) -> QueryState:
    """Generate a response using the AI provider."""
    logger.info("Generating response...")
    
    # Select the best query to use
    if state.get("should_skip_enhance", False):
        logger.info("Using original query as enhancement was skipped")
        final_query = state["user_query"]
    else:
        final_query = state.get("final_query") or state.get("enhanced_query") or state["user_query"]
    
    # Generate response using the provider
    try:
        from ..config.config import get_provider_config
        from ..providers import AIProvider
        
        provider_config = get_provider_config()
        provider = AIProvider.create(provider_config)
        
        logger.info(f"Sending query to provider: {final_query[:50]}...")
        response = await provider.generate_response(final_query)
        
        logger.info("Response generated successfully")
        return {**state, "final_query": final_query, "response": response}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {**state, "final_query": final_query, "response": f"Error generating response: {str(e)}"}

# Conditional routing functions
def route_after_match(state: QueryState) -> str:
    """Route based on whether enhancement should be skipped."""
    if state.get("should_skip_enhance", False):
        return "generate"
    else:
        return "enhance"

def route_based_on_validation(state: QueryState) -> str:
    """Route based on validation result."""
    if state.get("should_skip_enhance", False):
        return "generate"
    elif state["validation_result"] == "NEEDS_ADJUSTMENT":
        return "adjust"
    else:
        return "generate"

# Create the complete workflow
def create_workflow():
    """Create and return the LangGraph workflow."""
    # Create the graph with type hints
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("match_prompt", match_prompt)
    workflow.add_node("enhance", enhance_query)
    workflow.add_node("validate", validate_query)
    workflow.add_node("adjust", adjust_query)
    workflow.add_node("generate", generate_response)
    
    # Define edges
    workflow.add_edge(START, "match_prompt")
    workflow.add_conditional_edges(
        "match_prompt",
        route_after_match,
        {
            "enhance": "enhance",
            "generate": "generate"
        }
    )
    workflow.add_edge("enhance", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_based_on_validation,
        {
            "adjust": "adjust",
            "generate": "generate"
        }
    )
    workflow.add_edge("adjust", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph
    return workflow.compile()
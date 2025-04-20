"""
Core workflow engine for query enhancement and response generation.
"""
import logging
import json
import re
import asyncio
from typing import Dict, Any, TypedDict, List, Optional, Tuple

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage
from ..providers import AIProvider

# Set up logging
logger = logging.getLogger("core4ai.engine.workflow")

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
    provider_config: Dict[str, Any]  # Provider configuration to use consistently throughout workflow

# Define workflow nodes
async def match_prompt(state: QueryState) -> QueryState:
    """Match user query to the most appropriate prompt template."""
    import json
    import re
    from langchain_core.prompts import ChatPromptTemplate
    from ..engine.models import PromptMatch
    
    logger.info(f"Matching query to prompt template: {state['user_query']}")
    query = state['user_query']
    available_prompts = state.get('available_prompts', {})
    
    # Skip if no prompts available
    if not available_prompts:
        logger.warning("No prompts available, skipping enhancement")
        return {**state, "should_skip_enhance": True, "prompt_match": {"status": "no_prompts_available"}}

    # Create prompt details dictionary
    prompt_details = {}
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
            description = name.replace("_prompt", "").replace("_", " ")
        
        prompt_details[name] = {
            "variables": variables,
            "description": description
        }
    
    # Get provider
    provider_config = state.get('provider_config', {})
    provider = AIProvider.create(provider_config)
    
    # Create prompt template
    match_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a prompt matching assistant. Your task is to match a user query to the most 
            appropriate prompt template from a list of available templates. Choose the template 
            that best fits the user's intent and requirements."""
        ),
        (
            "user",
            """Match this user query to the most appropriate prompt template:
            
            User query: "{query}"
            
            Available templates:
            {templates}
            
            Choose the most appropriate template based on the intent and requirements.
            If none are appropriate, use "none" as the prompt_name.
            """
        )
    ])
    
    # Maximum attempts for retry
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Create chain with structured output
            structured_llm = provider.with_structured_output(PromptMatch, method='function_calling')
            match_chain = match_prompt_template | structured_llm
            
            # Invoke the chain with our variables
            match_result = await match_chain.ainvoke({
                "query": query,
                "templates": json.dumps(prompt_details, indent=2)
            })
            
            # Process the validated PromptMatch object
            prompt_name = match_result.prompt_name
            if prompt_name == "none":
                logger.info("No matching prompt found (LLM returned 'none')")
                return {
                    **state, 
                    "content_type": None,
                    "prompt_match": {
                        "status": "no_match",
                        "reasoning": match_result.reasoning
                    },
                    "should_skip_enhance": True
                }
            
            # Found a match with validated structure
            content_type = prompt_name.replace("_prompt", "")
            logger.info(f"Matched query to '{prompt_name}' with {match_result.confidence}% confidence")
            
            return {
                **state, 
                "content_type": content_type,
                "prompt_match": {
                    "status": "matched",
                    "prompt_name": prompt_name,
                    "confidence": match_result.confidence,
                    "reasoning": match_result.reasoning,
                },
                "parameters": match_result.parameters,
                "should_skip_enhance": False
            }
            
        except Exception as e:
            logger.warning(f"Structured match attempt {attempt}/{max_attempts} failed: {str(e)}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying structured matching (attempt {attempt+1}/{max_attempts})")
                continue
            
            # Final attempt failed, fall back to keyword matching
            logger.warning("All structured match attempts failed. Falling back to keyword matching.")
            
            # [Your existing keyword matching code remains here]
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
                
                # [Specific parameter extraction logic for different prompt types]
                
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
    """Validate that the enhanced query maintains the original intent."""
    from langchain_core.prompts import ChatPromptTemplate
    from ..engine.models import ValidationResult
    
    logger.info("Validating enhanced query...")
    
    # Skip validation if enhancement was skipped
    if state.get("should_skip_enhance", False):
        return {**state, "validation_result": "VALID", "validation_issues": []}
    
    user_query = state['user_query']
    enhanced_query = state['enhanced_query']
    
    # Get provider
    provider_config = state.get('provider_config', {})
    provider = AIProvider.create(provider_config)
    
    # Create prompt template
    validation_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a prompt validation assistant. Your task is to validate if an enhanced 
            prompt maintains the original user's intent and is well-formed. Check for issues 
            like missing key topics, repetitive text, or grammatical problems."""
        ),
        (
            "user",
            """Validate if this enhanced prompt maintains the original intent:
            
            Original query: "{original_query}"
            
            Enhanced prompt:
            {enhanced_query}
            
            Return whether the enhanced prompt is valid and a list of any issues found.
            """
        )
    ])
    
    # Maximum attempts for retry
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Create chain with structured output
            structured_llm = provider.with_structured_output(ValidationResult)
            validation_chain = validation_prompt_template | structured_llm
            
            # Invoke the chain with our variables
            validation_result = await validation_chain.ainvoke({
                "original_query": user_query,
                "enhanced_query": enhanced_query
            })
            
            # Get validation results from the validated object
            validation_issues = validation_result.issues if not validation_result.valid else []
            final_validation = "NEEDS_ADJUSTMENT" if validation_issues else "VALID"
            
            logger.info(f"Validation result: {final_validation}")
            return {**state, "validation_result": final_validation, "validation_issues": validation_issues}
            
        except Exception as e:
            logger.warning(f"Structured validation attempt {attempt}/{max_attempts} failed: {str(e)}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying structured validation (attempt {attempt+1}/{max_attempts})")
                continue
            
            # Final attempt failed, fall back to rule-based validation
            logger.warning("All structured validation attempts failed. Using rule-based validation.")
            
            # Rule-based validation fallback
            validation_issues = []
            
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
            
            # Check if major words from the original query are present
            major_words = [word for word in user_query.lower().split() 
                         if len(word) > 4 and word not in ["write", "about", "create", "make"]]
            
            missing_words = [word for word in major_words 
                           if word not in enhanced_query.lower()]
            
            if missing_words:
                validation_issues.append(f"Missing key words: {', '.join(missing_words)}")
            
            # Determine final validation result from rule-based validation
            final_validation = "NEEDS_ADJUSTMENT" if validation_issues else "VALID"
            
            logger.info(f"Rule-based validation result: {final_validation}")
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
        from ..providers import AIProvider
        
        # IMPORTANT: Use the provider_config from the state for consistency
        provider_config = state.get('provider_config', {})
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
        from ..providers import AIProvider
        
        # IMPORTANT: Use the provider_config from the state for consistency
        provider_config = state.get('provider_config', {})
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
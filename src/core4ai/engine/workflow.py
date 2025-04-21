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
    
    # Check if we're using Ollama and need to format differently
    provider_type = provider_config.get('type', '').lower()
    
    # Maximum attempts for retry
    max_attempts = 3
    last_error = None
    raw_responses = []
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Create system message with more guidance and examples
            system_message = """You are a prompt matching assistant. Your task is to match a user query to the most 
            appropriate prompt template from a list of available templates. Choose the template 
            that best fits the user's intent and requirements."""
            
            # Add format instructions based on provider type
            if provider_type == 'ollama':
                system_message += """
                
                IMPORTANT: You must respond with a valid JSON object in the exact format specified, 
                with no additional text before or after. The response must be parseable as JSON.
                """
            
            # Add feedback from previous errors to help the LLM correct its response
            if last_error and attempt > 1:
                system_message += f"""
                
                PREVIOUS ATTEMPT FAILED: {last_error}
                
                Your previous response could not be parsed correctly. Common issues include:
                - Missing or misspelled fields (prompt_name, confidence, reasoning, parameters)
                - Invalid JSON syntax (missing quotes, commas, braces)
                - Empty or null values for required fields
                - Text outside the JSON object
                
                Please ensure your response follows the required format.
                """
                
                if raw_responses:
                    system_message += f"""
                    
                    Your previous raw response was:
                    {raw_responses[-1]}
                    
                    Please fix the issues and provide a properly formatted response.
                    """
            
            # Customize user prompt based on provider type
            if provider_type == 'ollama':
                user_prompt = """Match this user query to the most appropriate prompt template:
                
                User query: "{query}"
                
                Available templates:
                {templates}
                
                RESPOND ONLY WITH JSON in this exact format (no other text before or after):
                
                {{
                    "prompt_name": "template_name_here",
                    "confidence": 85,
                    "reasoning": "Brief explanation of why this template is appropriate",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2"
                    }}
                }}
                
                RULES:
                - prompt_name MUST be exactly one of these values: {template_names} or "none"
                - confidence MUST be a number between 0-100
                - reasoning MUST be a string explaining your choice
                - parameters MUST be an object with values extracted from the query
                - Your entire response must be VALID JSON only
                
                EXAMPLES:
                
                VALID RESPONSE 1:
                {{
                    "prompt_name": "essay_prompt",
                    "confidence": 95,
                    "reasoning": "The query explicitly asks for an essay on a specific topic",
                    "parameters": {{
                        "topic": "artificial intelligence"
                    }}
                }}
                
                VALID RESPONSE 2:
                {{
                    "prompt_name": "none",
                    "confidence": 0,
                    "reasoning": "The query doesn't match any available template",
                    "parameters": {{}}
                }}
                """
            else:
                user_prompt = """Match this user query to the most appropriate prompt template:
                
                User query: "{query}"
                
                Available templates:
                {templates}
                
                Choose the most appropriate template based on the intent and requirements.
                If none are appropriate, use "none" as the prompt_name.
                
                For your reference, valid prompt names are: {template_names} or "none"
                
                Be sure to extract any relevant parameters from the query that would be needed
                to fill the template variables.
                """
            
            match_prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("user", user_prompt)
            ])
            
            # Capture the raw response before structured parsing for debugging
            raw_response = None
            
            # Create a custom chain that captures the raw output before parsing
            if provider_type == 'ollama':
                from langchain_core.runnables import RunnablePassthrough
                
                async def capture_raw_response(messages):
                    nonlocal raw_response
                    temp_model = provider.langchain_model
                    response = await temp_model.ainvoke(messages)
                    raw_response = response.content
                    return messages
                
                capture_chain = RunnablePassthrough() | capture_raw_response
                prompt_chain = match_prompt_template | capture_chain
                
                # Execute the chain to capture raw response
                messages = await prompt_chain.ainvoke({
                    "query": query,
                    "templates": json.dumps(prompt_details, indent=2),
                    "template_names": ", ".join([f'"{name}"' for name in available_prompts.keys()])
                })
                
                # Store the raw response for debugging
                if raw_response:
                    raw_responses.append(raw_response)
                    logger.debug(f"Raw LLM response (attempt {attempt}):\n{raw_response}")
                
                # For Ollama - custom handling if needed
                if provider_type == 'ollama':
                    # Try direct JSON parsing before using structured output
                    try:
                        # If we have a raw response, try to parse it directly
                        if raw_response:
                            # Clean up the response to handle potential text around JSON
                            import re
                            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                # Try to parse the JSON
                                json_data = json.loads(json_str)
                                
                                # Validate the parsed data has required fields
                                if all(k in json_data for k in ["prompt_name", "confidence", "reasoning", "parameters"]):
                                    # Create a PromptMatch object manually
                                    match_result = PromptMatch(
                                        prompt_name=json_data["prompt_name"],
                                        confidence=json_data["confidence"],
                                        reasoning=json_data["reasoning"],
                                        parameters=json_data["parameters"]
                                    )
                                    logger.info(f"Successfully parsed Ollama response directly as JSON")
                                else:
                                    # Missing required fields, continue to structured output
                                    logger.debug(f"JSON missing required fields, trying structured output")
                                    raise ValueError("JSON missing required fields")
                            else:
                                # No JSON found, continue to structured output
                                logger.debug(f"No JSON object found in response, trying structured output")
                                raise ValueError("No JSON object found in response")
                        else:
                            # No raw response, continue to structured output
                            raise ValueError("No raw response captured")
                    except Exception as json_err:
                        logger.debug(f"Direct JSON parsing failed: {json_err}, falling back to structured output")
                        # Now use the structured output chain
                        structured_llm = provider.with_structured_output(PromptMatch, method='function_calling')
                        match_result = await structured_llm.ainvoke(messages)
                else:
                    # Now use the structured output chain
                    structured_llm = provider.with_structured_output(PromptMatch, method='function_calling')
                    match_result = await structured_llm.ainvoke(messages)
            else:
                # For non-Ollama providers like OpenAI, use standard approach
                structured_llm = provider.with_structured_output(PromptMatch, method='function_calling')
                match_chain = match_prompt_template | structured_llm
                
                # Invoke the chain with our variables
                match_result = await match_chain.ainvoke({
                    "query": query,
                    "templates": json.dumps(prompt_details, indent=2),
                    "template_names": ", ".join([f'"{name}"' for name in available_prompts.keys()])
                })
            
            # Log the complete match result
            logger.info(f"📊 LLM MATCH RESPONSE (attempt {attempt}/{max_attempts}):")
            logger.info(f"  prompt_name: {match_result.prompt_name}")
            logger.info(f"  confidence: {match_result.confidence}")
            logger.info(f"  reasoning: {match_result.reasoning}")
            logger.info(f"  parameters: {json.dumps(match_result.parameters, indent=2) if match_result.parameters else '{}'}")
            
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
            
            # Validate the prompt_name is in available_prompts
            if prompt_name not in available_prompts:
                logger.warning(f"LLM returned invalid prompt name: {prompt_name}")
                
                # Try a similar prompt name if possible
                from difflib import get_close_matches
                close_matches = get_close_matches(prompt_name, available_prompts.keys(), n=1)
                if close_matches:
                    closest_match = close_matches[0]
                    logger.info(f"Using closest match: {closest_match}")
                    prompt_name = closest_match
                else:
                    logger.warning(f"No close match found, retrying")
                    raise ValueError(f"Invalid prompt name: {prompt_name}")
            
            # Found a match with validated structure
            content_type = prompt_name.replace("_prompt", "")
            
            # Add more detailed logging
            logger.info(f"✅ Matched query to '{prompt_name}' with {match_result.confidence}% confidence (attempt {attempt}/{max_attempts})")
            
            return {
                **state, 
                "content_type": content_type,
                "prompt_match": {
                    "status": "matched",
                    "prompt_name": prompt_name,
                    "confidence": match_result.confidence,
                    "reasoning": match_result.reasoning,
                    "fallback_used": False  # Standard LLM match, not a fallback
                },
                "parameters": match_result.parameters,
                "should_skip_enhance": False
            }
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"❌ Structured match attempt {attempt}/{max_attempts} failed: {last_error}")
            
            # Try to extract any partial JSON to understand what went wrong
            import traceback
            error_trace = traceback.format_exc()
            logger.debug(f"Full error traceback:\n{error_trace}")
            
            # Log the raw response if available
            if raw_response:
                logger.error(f"Raw response that caused the error (attempt {attempt}):\n{raw_response}")
            
            # If there's a JSON string in the error or raw response, try to extract and log it
            try:
                import re
                # First try to extract JSON from the raw response
                json_data = None
                if raw_response:
                    try:
                        # Try to parse the whole response as JSON
                        json_data = json.loads(raw_response)
                        logger.debug(f"Successfully parsed raw response as JSON:\n{json.dumps(json_data, indent=2)}")
                    except json.JSONDecodeError:
                        # If that fails, try to extract a JSON object using regex
                        json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            try:
                                json_data = json.loads(json_str)
                                logger.debug(f"Extracted JSON from raw response:\n{json.dumps(json_data, indent=2)}")
                            except:
                                logger.debug(f"Found JSON-like string but couldn't parse it:\n{json_str}")
                
                # If no JSON found in raw response, try the error message
                if not json_data:
                    json_match = re.search(r'(\{.*\})', str(e) + error_trace, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        try:
                            json_data = json.loads(json_str)
                            logger.debug(f"Extracted JSON from error:\n{json.dumps(json_data, indent=2)}")
                        except:
                            logger.debug(f"Found JSON-like string in error but couldn't parse it:\n{json_str}")
            except Exception as extract_err:
                logger.debug(f"Could not extract JSON: {extract_err}")
            
            if attempt < max_attempts:
                logger.info(f"⟳ Retrying with feedback (attempt {attempt+1}/{max_attempts})")
                continue
    
    # All attempts failed, fall back to keyword matching
    logger.warning("All structured match attempts failed. Falling back to keyword matching.")
    
    # Simple fallback to keyword matching
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
        "social_media": ["post", "tweet", "social media", "instagram", "facebook", "linkedin"],
        "report": ["report", "business report", "analysis report", "status", "findings"],
        "comparison": ["compare", "comparison", "versus", "vs", "differences", "similarities"],
        "research": ["research", "investigate", "study", "literature", "academic"],
        "interview": ["interview", "job", "hiring", "position", "application"],
        "sales_copy": ["sales", "copy", "product", "persuasive", "marketing"],
        "api_documentation": ["api", "documentation", "endpoint", "interface", "reference"],
        "syllabus": ["syllabus", "course", "lesson", "curriculum", "learning"]
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
        
        logger.info(f"✅ Matched query to '{prompt_name}' using fallback keyword matching")
        
        # Extract basic parameters based on content type
        parameters = {"topic": query.replace("write", "").replace("about", "").strip()}
        
        # Add more specific parameters for certain prompt types
        if content_type == "email":
            parameters["formality"] = "professional"
            parameters["recipient_type"] = "colleague"
        elif content_type == "creative":
            parameters["genre"] = "story"
        elif content_type == "code":
            parameters["language"] = "python"
            parameters["task"] = "implement a function to " + parameters["topic"]
        
        return {
            **state, 
            "content_type": content_type,
            "prompt_match": {
                "status": "matched",
                "prompt_name": prompt_name,
                "confidence": 70,  # Medium confidence for keyword matching
                "reasoning": f"Matched based on keywords in query",
                "fallback_used": True  # Flag that fallback was used
            },
            "parameters": parameters,
            "should_skip_enhance": False
        }
    
    # If no match found, skip enhancement
    logger.info("❌ No matching prompt found, skipping enhancement")
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
    import json
    
    logger.info("Validating enhanced query...")
    
    # Skip validation if enhancement was skipped
    if state.get("should_skip_enhance", False):
        logger.info("⏩ Skipping validation as enhancement was skipped")
        return {**state, "validation_result": "VALID", "validation_issues": []}
    
    user_query = state['user_query']
    enhanced_query = state['enhanced_query']
    
    # Get provider
    provider_config = state.get('provider_config', {})
    provider = AIProvider.create(provider_config)
    
    # Check if we're using Ollama and need to format differently
    provider_type = provider_config.get('type', '').lower()
    
    # Maximum attempts for retry
    max_attempts = 3
    last_error = None
    raw_responses = []
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Create system message with more guidance and examples
            system_message = """You are a prompt validation assistant. Your task is to validate if an enhanced 
            prompt maintains the original user's intent and is well-formed. Check for issues 
            like missing key topics, repetitive text, or grammatical problems."""
            
            # Add format instructions based on provider type
            if provider_type == 'ollama':
                system_message += """
                
                IMPORTANT: You must respond with a valid JSON object in the exact format specified, 
                with no additional text before or after. The response must be parseable as JSON.
                """
            
            # Add feedback from previous errors to help the LLM correct its response
            if last_error and attempt > 1:
                system_message += f"""
                
                PREVIOUS ATTEMPT FAILED: {last_error}
                
                Your previous response could not be parsed correctly. Common issues include:
                - Missing or misspelled fields (valid, issues)
                - Invalid JSON syntax (missing quotes, commas, braces)
                - Empty or null values for required fields
                - Text outside the JSON object
                
                Please ensure your response follows the required format.
                """
                
                if raw_responses:
                    system_message += f"""
                    
                    Your previous raw response was:
                    {raw_responses[-1]}
                    
                    Please fix the issues and provide a properly formatted response.
                    """
            
            # Customize user prompt based on provider type
            if provider_type == 'ollama':
                user_prompt = """Validate if this enhanced prompt maintains the original intent:
                
                Original query: "{original_query}"
                
                Enhanced prompt:
                {enhanced_query}
                
                RESPOND ONLY WITH JSON in this exact format (no other text before or after):
                
                {{
                    "valid": true_or_false,
                    "issues": [
                        "Issue 1 description",
                        "Issue 2 description"
                    ]
                }}
                
                RULES:
                - valid MUST be a boolean (true or false)
                - issues MUST be an array of strings (empty array if valid is true)
                - If valid is true, issues should be an empty array []
                - If valid is false, issues should contain at least one issue
                - Your entire response must be VALID JSON only
                
                EXAMPLES:
                
                VALID RESPONSE 1 (no issues):
                {{
                    "valid": true,
                    "issues": []
                }}
                
                VALID RESPONSE 2 (with issues):
                {{
                    "valid": false,
                    "issues": [
                        "Missing key topic from original query",
                        "Repetitive phrasing makes the prompt confusing"
                    ]
                }}
                """
            else:
                user_prompt = """Validate if this enhanced prompt maintains the original intent:
                
                Original query: "{original_query}"
                
                Enhanced prompt:
                {enhanced_query}
                
                Return whether the enhanced prompt is valid and a list of any issues found.
                Be specific about why the prompt is valid or invalid.
                """
            
            validation_prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("user", user_prompt)
            ])
            
            # Capture the raw response before structured parsing for debugging
            raw_response = None
            
            # Create a custom chain that captures the raw output before parsing
            if provider_type == 'ollama':
                from langchain_core.runnables import RunnablePassthrough
                
                async def capture_raw_response(messages):
                    nonlocal raw_response
                    temp_model = provider.langchain_model
                    response = await temp_model.ainvoke(messages)
                    raw_response = response.content
                    return messages
                
                capture_chain = RunnablePassthrough() | capture_raw_response
                prompt_chain = validation_prompt_template | capture_chain
                
                # Execute the chain to capture raw response
                messages = await prompt_chain.ainvoke({
                    "original_query": user_query,
                    "enhanced_query": enhanced_query
                })
                
                # Store the raw response for debugging
                if raw_response:
                    raw_responses.append(raw_response)
                    logger.debug(f"Raw LLM response (attempt {attempt}):\n{raw_response}")
                
                # For Ollama - custom handling if needed
                if provider_type == 'ollama':
                    # Try direct JSON parsing before using structured output
                    try:
                        # If we have a raw response, try to parse it directly
                        if raw_response:
                            # Clean up the response to handle potential text around JSON
                            import re
                            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                # Try to parse the JSON
                                json_data = json.loads(json_str)
                                
                                # Validate the parsed data has required fields
                                if all(k in json_data for k in ["valid", "issues"]):
                                    # Create a ValidationResult object manually
                                    validation_result = ValidationResult(
                                        valid=json_data["valid"],
                                        issues=json_data["issues"]
                                    )
                                    logger.info(f"Successfully parsed Ollama response directly as JSON")
                                else:
                                    # Missing required fields, continue to structured output
                                    logger.debug(f"JSON missing required fields, trying structured output")
                                    raise ValueError("JSON missing required fields")
                            else:
                                # No JSON found, continue to structured output
                                logger.debug(f"No JSON object found in response, trying structured output")
                                raise ValueError("No JSON object found in response")
                        else:
                            # No raw response, continue to structured output
                            raise ValueError("No raw response captured")
                    except Exception as json_err:
                        logger.debug(f"Direct JSON parsing failed: {json_err}, falling back to structured output")
                        # Now use the structured output chain
                        structured_llm = provider.with_structured_output(ValidationResult, method='function_calling')
                        validation_result = await structured_llm.ainvoke(messages)
                else:
                    # Now use the structured output chain
                    structured_llm = provider.with_structured_output(ValidationResult, method='function_calling')
                    validation_result = await structured_llm.ainvoke(messages)
            else:
                # For non-Ollama providers like OpenAI, use standard approach
                structured_llm = provider.with_structured_output(ValidationResult, method="function_calling")
                validation_chain = validation_prompt_template | structured_llm
                
                # Invoke the chain with our variables
                validation_result = await validation_chain.ainvoke({
                    "original_query": user_query,
                    "enhanced_query": enhanced_query
                })
            
            # Log the complete validation result
            logger.info(f"📊 LLM VALIDATION RESPONSE (attempt {attempt}/{max_attempts}):")
            logger.info(f"  valid: {validation_result.valid}")
            logger.info(f"  issues: {json.dumps(validation_result.issues, indent=2) if validation_result.issues else '[]'}")
            
            # Get validation results from the validated object
            validation_issues = validation_result.issues if not validation_result.valid else []
            final_validation = "NEEDS_ADJUSTMENT" if validation_issues else "VALID"
            
            # Improved logging with visual status
            if validation_issues:
                logger.info(f"❌ Validation result: {final_validation} with {len(validation_issues)} issues (attempt {attempt}/{max_attempts})")
                for i, issue in enumerate(validation_issues):
                    logger.info(f"  Issue {i+1}: {issue}")
            else:
                logger.info(f"✅ Validation result: VALID - Enhanced query maintains original intent (attempt {attempt}/{max_attempts})")
            
            return {**state, "validation_result": final_validation, "validation_issues": validation_issues}
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"❌ Structured validation attempt {attempt}/{max_attempts} failed: {last_error}")
            
            # Try to extract any partial JSON to understand what went wrong
            import traceback
            error_trace = traceback.format_exc()
            logger.debug(f"Full error traceback:\n{error_trace}")
            
            # Log the raw response if available
            if raw_response:
                logger.error(f"Raw response that caused the error (attempt {attempt}):\n{raw_response}")
            
            # If there's a JSON string in the error or raw response, try to extract and log it
            try:
                import re
                # First try to extract JSON from the raw response
                json_data = None
                if raw_response:
                    try:
                        # Try to parse the whole response as JSON
                        json_data = json.loads(raw_response)
                        logger.debug(f"Successfully parsed raw response as JSON:\n{json.dumps(json_data, indent=2)}")
                    except json.JSONDecodeError:
                        # If that fails, try to extract a JSON object using regex
                        json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            try:
                                json_data = json.loads(json_str)
                                logger.debug(f"Extracted JSON from raw response:\n{json.dumps(json_data, indent=2)}")
                            except:
                                logger.debug(f"Found JSON-like string but couldn't parse it:\n{json_str}")
                
                # If no JSON found in raw response, try the error message
                if not json_data:
                    json_match = re.search(r'(\{.*\})', str(e) + error_trace, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        try:
                            json_data = json.loads(json_str)
                            logger.debug(f"Extracted JSON from error:\n{json.dumps(json_data, indent=2)}")
                        except:
                            logger.debug(f"Found JSON-like string in error but couldn't parse it:\n{json_str}")
            except Exception as extract_err:
                logger.debug(f"Could not extract JSON: {extract_err}")
            
            if attempt < max_attempts:
                logger.info(f"⟳ Retrying with feedback (attempt {attempt+1}/{max_attempts})")
                continue
    
    # All attempts failed, fall back to rule-based validation
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
    
    # Log the fallback validation result with details
    if validation_issues:
        logger.info(f"⚠️ Rule-based validation result: {final_validation} with {len(validation_issues)} issues")
        for i, issue in enumerate(validation_issues):
            logger.info(f"  Issue {i+1}: {issue}")
    else:
        logger.info("✅ Rule-based validation result: VALID - No issues detected")
    
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
        
        # Use higher temperature for creative generation
        response = await provider.generate_response(final_query, temperature=0.7)
        
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
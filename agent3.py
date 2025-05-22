# agent3.py — Reasoner Agent
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional, Union, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GMILLM(Runnable):
    """Custom wrapper for GMI API that implements LangChain's Runnable interface"""
    model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', self.model_name)
        self.temperature = kwargs.get('temperature', self.temperature)
        self.max_tokens = kwargs.get('max_tokens', self.max_tokens)
        self.top_p = kwargs.get('top_p', self.top_p)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the GMI Cloud API with the given prompt, falling back to Claude 3.5 Sonnet if needed.
        
        Args:
            prompt: The prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional arguments
            
        Returns:
            The model's response as a string
        """
        import requests
        import re
        
        # Get API keys from environment
        gmi_api_key = os.getenv('GMI_API_KEY')
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Prepare mock response for when no API keys are available
        mock_response = {
            "optimized_schedule": [
                {
                    "task_id": "task1",
                    "start_time": "09:00",
                    "end_time": "10:00",
                    "location": "Work"
                },
                {
                    "task_id": "task2",
                    "start_time": "10:30",
                    "end_time": "11:30",
                    "location": "Grocery Store"
                }
            ],
            "feedback": "This is a mock response. Please set either GMI_API_KEY or ANTHROPIC_API_KEY to use the real service.",
            "adjustments": {
                "total_travel_time_saved": 0,
                "constraints_violated": [],
                "preferences_met": []
            }
        }
        
        # If no API keys are available, return mock response
        if not gmi_api_key and not anthropic_api_key:
            print("Warning: Neither GMI_API_KEY nor ANTHROPIC_API_KEY is set, using mock response")
            return json.dumps(mock_response)
            
        # Try GMI Cloud first if API key is available
        if gmi_api_key:
            try:
                print("Attempting to use GMI Cloud API...")
                # Prepare the API request
                headers = {
                    "Authorization": f"Bearer {gmi_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Prepare the payload with model and messages
                payload = {
                    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that helps with schedule optimization. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": kwargs.get('temperature', self.temperature),
                    "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "response_format": {"type": "json_object"}
                }
                
                # Add stop words if provided
                if stop:
                    payload["stop"] = stop
                
                # Make the API request
                response = requests.post(
                    "https://api.gmi-serving.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10  # Reduced from 30 to 10 seconds
                )
                response.raise_for_status()
                
                # Parse the response
                response_json = response.json()
                print(f"GMI API Raw Response: {json.dumps(response_json, indent=2)}")
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    choice = response_json["choices"][0]
                    
                    # Extract content based on response format
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    elif "text" in choice:
                        content = choice["text"]
                    elif "content" in choice:
                        content = choice["content"]
                    else:
                        print("Unexpected response format - no message, text, or content in choices")
                        raise ValueError("Unexpected response format from GMI API")
                    
                    print(f"Extracted content: {content}")
                    
                    # Try to parse the content as JSON if it's a string
                    try:
                        if isinstance(content, str):
                            # Try to extract JSON from markdown code blocks
                            json_match = re.search(r'```(?:json\n)?({.*?})```', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1).strip()
                                print(f"Extracted JSON from markdown: {json_str}")
                                return json_str
                            
                            # If no markdown, try to find JSON object directly
                            json_match = re.search(r'({.*})', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1).strip()
                                print(f"Extracted JSON from text: {json_str}")
                                return json_str
                            
                            # If still no JSON, try to parse the entire content
                            print("Trying to parse entire content as JSON")
                            return content
                        
                        # If content is already a dict, return it as JSON string
                        if isinstance(content, dict):
                            return json.dumps(content)
                            
                    except (json.JSONDecodeError, AttributeError) as je:
                        print(f"Failed to parse JSON content: {str(je)}")
                        print(f"Content that failed to parse: {content}")
                        raise ValueError(f"Failed to parse JSON response: {str(je)}")
                    
                    return content
                
                print("No choices in response")
                raise ValueError("No choices in response from GMI API")
                
            except Exception as e:
                print(f"Error with GMI API, falling back to Claude: {str(e)}")
                # Fall through to Claude if GMI fails
        
        # If we get here, either GMI failed or we're using Claude as primary
        if anthropic_api_key:
            try:
                print("Falling back to Anthropic Claude 3.5 Sonnet...")
                try:
                    from anthropic import Anthropic
                except ImportError:
                    print("anthropic package not found. Installing...")
                    import sys
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
                    from anthropic import Anthropic
                
                client = Anthropic(api_key=anthropic_api_key)
                
                # Format the prompt for Claude
                system_prompt = """You are a helpful assistant that helps with schedule optimization. 
                Please respond with a valid JSON object containing an optimized schedule.
                The response should include:
                - optimized_schedule: List of scheduled tasks with task_id, start_time, end_time, and location
                - feedback: A brief explanation of the optimization
                - adjustments: Any adjustments made to the schedule
                
                Only return the JSON object, no other text or markdown formatting."""
                
                try:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        temperature=0.7,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Extract and clean the response
                    content = response.content[0].text.strip()
                    print(f"Claude raw response: {content[:500]}...")  # Print first 500 chars to avoid huge logs
                    
                    # Try to extract JSON if it's wrapped in markdown code blocks
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    # Validate the JSON
                    try:
                        parsed = json.loads(content)
                        print("Successfully parsed Claude response as JSON")
                        return json.dumps(parsed)  # Return as string to be consistent
                    except json.JSONDecodeError as je:
                        print(f"Failed to parse Claude response as JSON: {str(je)}")
                        print(f"Content that failed to parse: {content[:500]}...")
                        # Try to extract just the JSON object if possible
                        try:
                            json_match = re.search(r'({.*})', content, re.DOTALL)
                            if json_match:
                                print("Extracted JSON from response")
                                return json_match.group(1).strip()
                        except Exception as e:
                            print(f"Error extracting JSON from response: {str(e)}")
                        
                        # If we can't parse the response, raise an error
                        raise ValueError(f"Failed to parse Claude response as JSON: {str(je)}")
                    
                except Exception as api_error:
                    print(f"Error calling Claude API: {str(api_error)}")
                    raise
                
            except Exception as e:
                print(f"Error initializing Claude client: {str(e)}")
                # Fall through to mock response
        
        # If we get here, all API calls failed
        print("All API calls failed, using mock response")
        return json.dumps(mock_response)
    
    # Required for Runnable interface
    def invoke(self, input: Union[str, Dict, Any], config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        # Handle case where input is a dictionary with a 'prompt' key
        if isinstance(input, dict) and 'prompt' in input:
            prompt = input['prompt']
            stop = input.get('stop')
            return self._call(prompt, stop=stop, **kwargs)
        # Handle case where input is a string
        elif isinstance(input, str):
            stop = kwargs.pop('stop', None) if kwargs else None
            return self._call(input, stop=stop, **kwargs)
        # Handle case where input is a StringPromptValue
        elif hasattr(input, 'to_string'):
            stop = kwargs.pop('stop', None) if kwargs else None
            return self._call(input.to_string(), stop=stop, **kwargs)
        # Handle case where input is a list of messages (e.g., ChatPromptValue)
        elif hasattr(input, 'to_messages'):
            stop = kwargs.pop('stop', None) if kwargs else None
            messages = input.to_messages()
            # Convert messages to a single string
            prompt = "\n".join(f"{msg.type}: {msg.content}" if hasattr(msg, 'type') else str(msg) for msg in messages)
            return self._call(prompt, stop=stop, **kwargs)
        else:
            # Try to convert the input to a string as a last resort
            try:
                stop = kwargs.pop('stop', None) if kwargs else None
                return self._call(str(input), stop=stop, **kwargs)
            except Exception as e:
                raise ValueError(f"Unsupported input type: {type(input)} - {str(e)}")
    
    # Required for LLM compatibility
    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> LLMResult:
        from langchain_core.outputs import Generation, LLMResult
        
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])
            
        return LLMResult(generations=generations)
    
    # For backward compatibility with older LangChain versions
    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self._call(prompt, stop=stop, **kwargs)
        
    # Add this method to make it work with newer LangChain versions
    async def _ainvoke(self, input: Union[str, Dict], **kwargs) -> Any:
        return self.invoke(input, **kwargs)
        
    # Add this method to make it work with newer LangChain versions
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # For simplicity, just call the synchronous version
        return self.generate(prompts, stop=stop, **kwargs)

class ScheduleOptimizationRequest(BaseModel):
    tasks: List[Dict[str, Any]]
    locations: List[Dict[str, Any]]
    travel_info: List[Dict[str, Any]]
    constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}

class ScheduleOptimizationResponse(BaseModel):
    optimized_schedule: List[Dict[str, Any]]
    feedback: str
    adjustments: Dict[str, Any] = {}

def find_location_name(location_id: str, locations: List[Dict[str, Any]]) -> str:
    """Helper function to find a location name by ID"""
    for loc in locations:
        if loc.get('id') == location_id:
            return loc.get('name', 'Unknown')
    return 'Unknown'

def run_reasoner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the reasoner agent on the current state.
    
    Args:
        state: Current system state containing tasks, locations, and travel info
        
    Returns:
        Updated state with optimization results
    """
    print("\n=== Running Reasoner Agent ===")
    
    # Extract tasks, locations, etc.
    tasks = state.get('tasks', [])
    locations = state.get('locations', [])
    travel_info = state.get('travel_info', [])
    constraints = state.get('constraints', {})
    preferences = state.get('preferences', {})
    
    # Prepare task, location, and constraint information for the prompt
    task_info = []
    for task in tasks:
        task_info.append(f"- {task['name']} (ID: {task['id']}): {task['description']} "
                       f"at {task.get('location', {}).get('name', 'Unknown')} "
                       f"for {task.get('duration_minutes', 0)} minutes")
    
    location_info = []
    for loc in locations:
        location_info.append(f"- {loc['name']} (ID: {loc['id']}): {loc.get('address', 'No address')}")
    
    travel_info_str = []
    for travel in travel_info:
        from_loc = find_location_name(travel['from_location_id'], locations)
        to_loc = find_location_name(travel['to_location_id'], locations)
        travel_info_str.append(f"- From {from_loc} to {to_loc}: {travel.get('travel_time_minutes', 0)} minutes")
    
    # Prepare the prompt for the LLM
    task_list_str = '\n'.join(task_info) if task_info else "No tasks provided"
    location_list_str = '\n'.join(location_info) if location_info else "No locations provided"
    travel_times_str = '\n'.join(travel_info_str) if travel_info_str else "No travel info provided"
    constraint_list_str = '\n'.join([f"- {k}: {v}" for k, v in constraints.items()]) if constraints else "No constraints provided"
    preference_list_str = '\n'.join([f"- {k}: {v}" for k, v in preferences.items()]) if preferences else "No preferences provided"
    
    prompt = f"""
    You are a scheduling optimization expert. Your task is to optimize the following schedule 
    based on the given tasks, locations, and constraints.
    
    TASKS:
    {task_list_str}
    
    LOCATIONS:
    {location_list_str}
    
    TRAVEL TIMES BETWEEN LOCATIONS:
    {travel_times_str}
    
    CONSTRAINTS:
    {constraint_list_str}
    
    PREFERENCES:
    {preference_list_str}
    
    Please provide an optimized schedule that respects all constraints and preferences.
    Return ONLY a valid JSON object with the following structure (no markdown formatting, just pure JSON):
    {{
        "optimized_schedule": [
            {{
                "task_id": "task1",
                "start_time": "HH:MM",
                "end_time": "HH:MM",
                "location": "Location Name"
            }}
        ],
        "feedback": "Explanation of optimizations made",
        "adjustments": {{
            "total_travel_time_saved": 0,
            "constraints_violated": [],
            "preferences_met": []
        }}
    }}
    """
    
    print("\n=== Sending to LLM ===")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Initialize the GMI LLM
    llm = GMILLM()
    
    try:
        # Call the GMI LLM
        print("\nCalling GMI API...")
        response = llm._call(prompt)
        print(f"\n=== Raw API Response ===\n{response}\n")
        
        # If response is already a dict, use it directly
        if isinstance(response, dict):
            result = response
        # If response is a string, try to parse it as JSON
        elif isinstance(response, str):
            try:
                # Clean up the response string
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Parse the JSON
                result = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"Failed to parse response as JSON: {e}")
                print(f"Response content: {response}")
                state['optimization_feedback'] = f"Error parsing optimization response: {e}"
                state['optimization_adjustments'] = {"error": str(e)}
                state['current_agent'] = 'reasoner'
                state['converged'] = True
                return state
        else:
            raise ValueError(f"Unexpected response type: {type(response).__name__}")
        
        # Update the state with the optimized schedule
        if 'optimized_schedule' in result:
            state['optimized_schedule'] = result['optimized_schedule']
            state['optimization_feedback'] = result.get('feedback', 'Optimization completed successfully')
            state['optimization_adjustments'] = result.get('adjustments', {})
            print("Optimization completed successfully")
            print(f"Optimized schedule: {json.dumps(result['optimized_schedule'], indent=2)}")
        else:
            state['optimization_feedback'] = "Optimization response missing 'optimized_schedule' field"
            state['optimization_adjustments'] = {"error": "Missing optimized_schedule in response"}
            print(f"Unexpected response format: {result}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        state['optimization_feedback'] = f"Error during optimization: {e}"
        state['optimization_adjustments'] = {"error": str(e)}
    
    # Update the current agent and mark as converged
    state['current_agent'] = 'reasoner'
    state['converged'] = True
    
    return state

def optimize_schedule(request: ScheduleOptimizationRequest) -> ScheduleOptimizationResponse:
    """
    Optimize the schedule using LLM-based reasoning.
    
    Args:
        request: Schedule optimization request containing tasks, locations, and travel info
        
    Returns:
        ScheduleOptimizationResponse with optimized schedule and feedback
    """
    # Prepare the prompt for the LLM
    prompt_template = """You are an expert schedule optimizer. Your task is to analyze the following schedule and optimize it based on the given constraints and preferences.

Current Schedule:
{tasks}

Locations:
{locations}

Travel Information:
{travel_info}

Constraints:
{constraints}

Preferences:
{preferences}

Please analyze this schedule and provide an optimized version that:
1. Respects all time constraints
2. Minimizes travel time between locations
3. Takes into account the provided preferences
4. Handles any conflicts or issues

Return your response as a JSON object with the following structure:
{{
  "optimized_schedule": [
    {{
      "task_id": "id",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "location": "location_name"
    }}
  ],
  "feedback": "Your analysis and reasoning",
  "adjustments": {{
    "total_travel_time_saved": 30,
    "constraints_violated": [],
    "preferences_met": ["morning_meetings", "afternoon_focus_time"]
  }}
}}"""

    # Format the input data
    tasks_str = "\n".join([
        f"- {task['name']} (ID: {task['id']}): {task.get('start_time', '')} - {task.get('end_time', '')} at {task.get('location', {}).get('name', 'unknown')}"
        for task in request.tasks
    ])
    
    locations_str = "\n".join([
        f"- {loc['name']} (ID: {loc.get('task_id', 'N/A')}): {loc.get('address', 'No address')} "
        f"({loc.get('lat', 'N/A')}, {loc.get('lng', 'N/A')})"
        for loc in request.locations
    ])
    
    travel_info_str = "\n".join([
        f"- From {trip['from_task_id']} to {trip['to_task_id']}: {trip.get('travel_time_minutes', 0)} min, {trip.get('distance_km', 0):.1f} km"
        for trip in request.travel_info
    ])
    
    constraints_str = "\n".join([f"- {k}: {v}" for k, v in request.constraints.items()])
    preferences_str = "\n".join([f"- {k}: {v}" for k, v in request.preferences.items()])
    
    # Create and run the LLM chain
    llm = GMILLM()
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["tasks", "locations", "travel_info", "constraints", "preferences"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run(
            tasks=tasks_str,
            locations=locations_str,
            travel_info=travel_info_str,
            constraints=constraints_str or "None",
            preferences=preferences_str or "None"
        )
        
        # Parse the LLM response
        response_data = json.loads(result.strip())
        
        # Ensure the response has all required fields
        if 'schedule' not in response_data:
            response_data['schedule'] = []
        if 'feedback' not in response_data:
            response_data['feedback'] = 'Optimization completed successfully'
        if 'adjustments' not in response_data:
            response_data['adjustments'] = {}
        
        # Ensure each schedule item has all required fields
        for item in response_data['schedule']:
            item.setdefault('task_id', '')
            item.setdefault('start_time', '09:00')
            item.setdefault('end_time', '10:00')
            item.setdefault('location_id', 'unknown')
        
        # Add converged flag to indicate if optimization is complete
        response_data['converged'] = True
        
        return response_data
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return {
            'schedule': [],
            'feedback': f'Error during optimization: {str(e)}',
            'adjustments': {'error': str(e)},
            'converged': True  # Stop further iterations on error
        }

    # Extract and validate tasks
    tasks = state.get("tasks", [])
    if not tasks:
        return {
            **state,
            "error": "No tasks provided for optimization",
            "optimized_schedule": [],
            "feedback": "No tasks to optimize"
        }
    
    # Extract and validate locations
    locations = state.get("locations", [])
    if not locations:
        return {
            **state,
            "error": "No locations provided for optimization",
            "optimized_schedule": [],
            "feedback": "No location data available"
        }
    
    # Extract and validate travel info
    travel_info = state.get("travel_info", [])
    if not travel_info:
        print("Warning: No travel information provided, using default values")
        # Generate default travel info if none provided
        travel_info = []
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    travel_info.append({
                        'from_task_id': f'task_{i+1}',
                        'to_task_id': f'task_{j+1}',
                        'from_location_id': f'loc{i+1}',
                        'to_location_id': f'loc{j+1}',
                        'distance_km': 10,
                        'travel_time_min': 15,
                        'travel_time_minutes': 15,
                        'mode': 'driving'
                    })
    
    # Create the optimization request
    request = ScheduleOptimizationRequest(
        tasks=tasks,
        locations=locations,
        travel_info=travel_info,
        constraints=state.get("constraints", {}),
        preferences=state.get("preferences", {})
    )
    
    # Run optimization
    response = optimize_schedule(request)
    
    # Update state with results
    return {
        **state,
        "optimized_schedule": response.get('schedule', []),
        "optimization_feedback": response.get('feedback', 'No feedback provided'),
        "optimization_adjustments": response.get('adjustments', {}),
        "current_agent": "reasoner",
        "agent_messages": {
            **state.get("agent_messages", {}),
            "reasoner": [response.get('feedback', 'No feedback provided')]
        },
        "converged": response.get('converged', True)
    }

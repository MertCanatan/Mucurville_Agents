# agent3.py — Reasoner Agent
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Iterator
from pydantic import BaseModel, Field, field_validator
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.outputs import LLMResult
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
        Call the GMI Cloud API with the given prompt.
        
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
        
        # Get API key from environment
        api_key = os.getenv('GMI_API_KEY')
        
        # Prepare mock response for when API key is not available
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
            "feedback": "This is a mock response. Please set GMI_API_KEY to use the real service.",
            "adjustments": {
                "total_travel_time_saved": 0,
                "constraints_violated": [],
                "preferences_met": []
            }
        }
        
        if not api_key:
            print("Warning: GMI_API_KEY not set, using mock response")
            return json.dumps(mock_response)
            
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the payload with model and messages
        payload = {
            "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that helps with schedule optimization."},
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
        
        try:
            # Make the API request
            response = requests.post(
                "https://api.gmi-serving.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
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
                    return json.dumps(mock_response)
                
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
                    # Return mock response if parsing fails
                    return json.dumps(mock_response)
                
                return content
            
            print("No choices in response")
            return json.dumps(mock_response)
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling GMI API: {str(e)}")
            # Return mock response on error
            return json.dumps(mock_response)
        except json.JSONDecodeError as je:
            print(f"Failed to parse JSON response: {str(je)}")
            print(f"Response text: {response.text}")
            # Return mock response on JSON decode error
            return json.dumps(mock_response)
        except Exception as e:
            print(f"Error processing API response: {str(e)}")
            # Return mock response on any other error
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
    
    # Call the GMI LLM
    print("\nCalling GMI API...")
    try:
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
                
                # Parse the JSON response
                try:
                    result = json.loads(cleaned_response)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse response as JSON: {e}")
                    print(f"Response content: {response}")
                    result = {}
            except Exception as e:
                print(f"Error processing response: {e}")
                result = {}
        else:
            print(f"Unexpected response type: {type(response)}")
            result = {}
            
        # If we have a valid result with optimized_schedule
        if result and 'optimized_schedule' in result:
            optimized_schedule = result['optimized_schedule']
            
            # Ensure all tasks are included in the schedule
            scheduled_task_ids = {item['task_id'] for item in optimized_schedule if 'task_id' in item}
            all_task_ids = {task['id'] for task in tasks if 'id' in task}
            missing_task_ids = all_task_ids - scheduled_task_ids
            
            if missing_task_ids:
                print(f"Warning: Some tasks were not scheduled: {missing_task_ids}")
                # Add missing tasks to the schedule with default times
                current_time = datetime.strptime("09:00", "%H:%M").time()
                
                for task_id in missing_task_ids:
                    task = next((t for t in tasks if t.get('id') == task_id), None)
                    if task:
                        duration = task.get('duration_minutes', 60)
                        location = task.get('location', {}).get('name', 'Unknown Location')
                        
                        # Calculate end time
                        start_dt = datetime.combine(datetime.today(), current_time)
                        end_dt = start_dt + timedelta(minutes=duration)
                        
                        optimized_schedule.append({
                            'task_id': task_id,
                            'start_time': current_time.strftime("%H:%M"),
                            'end_time': end_dt.time().strftime("%H:%M"),
                            'location': location
                        })
                        
                        # Update current time for next task (add 15 min break)
                        current_time = (end_dt + timedelta(minutes=15)).time()
                        
                        print(f"Added missing task to schedule: {task['name']} at {location}")
            
            state['optimized_schedule'] = optimized_schedule
            state['optimization_feedback'] = result.get('feedback', 'Optimization completed with fallback scheduling')
            state['optimization_adjustments'] = result.get('adjustments', {})
            
            # Debug: Print the optimized schedule
            print("\n=== Debug: Optimized Schedule ===")
            print(f"Type: {type(optimized_schedule)}")
            if isinstance(optimized_schedule, list):
                print(f"Length: {len(optimized_schedule)}")
                if optimized_schedule:
                    print(f"First item type: {type(optimized_schedule[0])}")
                    print(f"First item keys: {optimized_schedule[0].keys() if hasattr(optimized_schedule[0], 'keys') else 'N/A'}")
            
            # Convert the schedule to the expected format
            if isinstance(optimized_schedule, list):
                for item in optimized_schedule:
                    if isinstance(item, dict) and 'task_id' in item and 'start_time' in item and 'end_time' in item:
                        task_id = item['task_id']
                        start_time = item['start_time']
                        end_time = item['end_time']
                        location = item.get('location', 'Unknown Location')
                        print(f"Added schedule item: Task {task_id} at {location}")
            
            # Update the current agent and mark as converged
            state['current_agent'] = 'reasoner'
            state['converged'] = True
            return state
            
    except Exception as e:
        print(f"Error calling GMI API: {e}")
        
    # Fallback to a simple scheduling algorithm if API call fails or no valid schedule
    print("\n=== Using fallback scheduling ===")
    optimized_schedule = []
    
    try:
        # Debug print to verify datetime import
        print(f"Debug - Current datetime: {datetime.now()}")
        print(f"Debug - Current date: {datetime.today()}")
        
        current_time = datetime.strptime("09:00", "%H:%M").time()
        print(f"Debug - Parsed start time: {current_time}")
        
        # Sort tasks by priority (high first) and then by duration (shortest first)
        sorted_tasks = sorted(
            [t for t in tasks if t.get("id")],
            key=lambda x: (
                0 if x.get("priority") == "high" else 1 if x.get("priority") == "medium" else 2,
                x.get("duration_minutes", 0)
            )
        )
        
        print(f"Debug - Sorted tasks: {[t['id'] for t in sorted_tasks]}")
        
        # Simple round-robin scheduling
        for task in sorted_tasks:
            task_id = task["id"]
            duration = task.get("duration_minutes", 30)
            location = task.get("location", {}).get("name", "Unknown Location")
            
            # Debug print before datetime operations
            print(f"Debug - Processing task {task_id} with duration {duration} minutes")
            print(f"Debug - Current time before processing: {current_time}")
            
            # Combine with today's date for timedelta operations
            start_dt = datetime.combine(datetime.today(), current_time)
            end_dt = start_dt + timedelta(minutes=duration)
            
            # Format times for the schedule
            start_time_str = current_time.strftime("%H:%M")
            end_time_str = end_dt.time().strftime("%H:%M")
            
            print(f"Debug - Scheduling task {task_id} from {start_time_str} to {end_time_str} at {location}")
            
            optimized_schedule.append({
                "task_id": task_id,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "location": location
            })
            
            # Add 15-minute break between tasks
            current_time = (end_dt + timedelta(minutes=15)).time()
            print(f"Debug - Next task will start at: {current_time}")
        
        state['optimized_schedule'] = optimized_schedule
        state['optimization_feedback'] = "Used fallback scheduling. All tasks have been scheduled with a simple round-robin algorithm."
        state['optimization_adjustments'] = {
            "total_travel_time_saved": 0,
            "constraints_violated": ["api_unavailable"],
            "preferences_met": []
        }
        
    except Exception as e:
        print(f"Error in fallback scheduling: {e}")
        import traceback
        traceback.print_exc()
        
        # Set a default schedule if there's an error
        state['optimized_schedule'] = []
        state['optimization_feedback'] = f"Error in fallback scheduling: {str(e)}"
        state['optimization_adjustments'] = {
            "error": str(e),
            "constraints_violated": ["scheduling_error"],
            "preferences_met": []
        }
    
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

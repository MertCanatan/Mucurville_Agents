# agent1.py — Planner Agent using LangChain
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.llms import OpenAI  # Using OpenAI as fallback
import json
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

def parse_tasks(task_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse and validate task input.
    
    Args:
        task_input: List of task dictionaries
        
    Returns:
        List of parsed and validated tasks
    """
    parsed_tasks = []
    for i, task in enumerate(task_input):
        if not isinstance(task, dict):
            print(f"Warning: Task at index {i} is not a dictionary, skipping")
            continue
            
        # Ensure required fields are present
        task_id = task.get('id', f'task_{i+1}')
        name = task.get('name', f'Task {i+1}')
        duration = task.get('duration_minutes', 60)  # Default to 60 minutes
        
        # Create parsed task with defaults
        parsed_task = {
            'id': task_id,
            'name': name,
            'description': task.get('description', ''),
            'duration_minutes': duration,
            'location_id': task.get('location_id', 'unknown'),
            'priority': task.get('priority', 'medium').lower(),
            'constraints': task.get('constraints', {}),
            'category': task.get('category', 'other')
        }
        
        parsed_tasks.append(parsed_task)
    
    return parsed_tasks

def apply_constraints(tasks: List[Dict[str, Any]], constraints: Dict[str, Any], preferences: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Apply constraints and preferences to tasks.
    
    Args:
        tasks: List of task dictionaries
        constraints: Dictionary of constraints to apply
        preferences: Optional dictionary of user preferences
        
    Returns:
        List of tasks with constraints and preferences applied
    """
    if not constraints and not preferences:
        return tasks
        
    for task in tasks:
        # Apply time window constraints if specified
        if 'time_windows' in constraints:
            if 'constraints' not in task:
                task['constraints'] = {}
            task['constraints']['time_windows'] = constraints['time_windows']
            
        # Apply priority constraints if specified
        if 'min_priority' in constraints:
            priority_order = {'low': 0, 'medium': 1, 'high': 2}
            min_priority = constraints['min_priority'].lower()
            task_priority = task.get('priority', 'medium').lower()
            if priority_order.get(task_priority, 0) < priority_order.get(min_priority, 0):
                task['priority'] = min_priority
                
        # Apply preferences if provided
        if preferences:
            # Example: Apply preferred time windows if specified
            if 'preferred_hours' in preferences:
                if 'constraints' not in task:
                    task['constraints'] = {}
                if 'time_windows' not in task['constraints']:
                    task['constraints']['time_windows'] = []
                task['constraints']['time_windows'].extend(preferences['preferred_hours'])
                
            # Apply any other preferences here
            if 'preferred_categories' in preferences and 'category' in task:
                if task['category'] in preferences['preferred_categories']:
                    # Boost priority for preferred categories
                    task['priority'] = max(
                        task.get('priority', 'medium'),
                        preferences['preferred_categories'][task['category']],
                        key=lambda x: {'low': 0, 'medium': 1, 'high': 2}.get(x, 0)
                    )
    
    return tasks

class GMILLM(LLM):
    """Custom LLM wrapper for GMI Cloud API with fallback to local model"""
    def _call(self, prompt, stop=None, run_manager=None):
        import json
        import requests
        from langchain.llms import OpenAI
        
        # First, try to use the GMI Cloud API if the key is available
        api_key = os.getenv("GMI_API_KEY")
        if api_key:
            try:
                # GMI Cloud API endpoint from documentation
                url = "https://api.gmi-serving.com/v1/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Format the payload according to GMI Cloud API documentation
                payload = {
                    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that helps with task planning and prioritization."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.95,
                    "stream": False,
                    "response_format": {"type": "json_object"}  # Ensure JSON response
                }
                
                print(f"Sending request to GMI Cloud API: {url}")
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                response_json = response.json()
                
                print(f"API Response: {json.dumps(response_json, indent=2)}")
                
                # Extract the content from the response
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    print(f"Extracted content: {content}")
                    # Try to parse the content as JSON
                    try:
                        # First, try to parse the content directly
                        parsed = json.loads(content)
                        # If the response has a "tasks" key, extract that
                        if isinstance(parsed, dict) and "tasks" in parsed:
                            return json.dumps(parsed["tasks"])
                        return content
                    except json.JSONDecodeError:
                        print("Response is not valid JSON, trying to extract JSON from response")
                        # Try to extract JSON array from the response
                        try:
                            json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                            if json_match:
                                return json_match.group(0)
                            return content
                        except Exception as e:
                            print(f"Error extracting JSON from response: {str(e)}")
                            return content
                return "[]"  # Return empty array as fallback
                
            except Exception as e:
                print(f"GMI Cloud API call failed: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}")
                    try:
                        print(f"Response body: {e.response.text}")
                    except:
                        print("Could not read response body")
        
        # Fallback to OpenAI if available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                print("Falling back to OpenAI API...")
                llm = OpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo-instruct")
                return llm(prompt)
            except Exception as e:
                print(f"OpenAI API call failed: {str(e)}")
        
        # Final fallback - return a hardcoded response
        print("Using hardcoded fallback response")
        return """
        [
            {
                "id": "task1",
                "category": "meeting",
                "fixed": true,
                "priority": "high"
            },
            {
                "id": "task2",
                "category": "errand",
                "fixed": false,
                "priority": "medium"
            },
            {
                "id": "task3",
                "category": "health",
                "fixed": false,
                "priority": "high"
            }
        ]
        """

    @property
    def _llm_type(self):
        return "gmi-custom-llm"

def plan_day(raw_task_input, df_d=None, df_p=None):
    """
    Planner Agent using LangChain for prioritization
    """

    # Step 1: Parse
    task_list = parse_tasks(raw_task_input)
    if not task_list:
        raise ValueError("No valid tasks found.")

    # Step 2: Constraints
    constrained = apply_constraints(task_list, df_d, df_p)

    # Step 3: LLM Prioritization via LangChain
    prompt_template = PromptTemplate(
        input_variables=["tasks", "preferences"],
        template=(
            "You are a helpful planner.\n"
            "Tasks:\n{tasks}\n\n"
            "Preferences:\n{preferences}\n\n"
            "Classify each task into a category (errand, meeting, fun, etc), "
            "label as Fixed or Flexible, and assign a priority (High, Medium, Low).\n"
            "Return as JSON."
        )
    )

    llm = GMILLM()
    chain = LLMChain(llm=llm, prompt=prompt_template)

    formatted_tasks = str(constrained)
    formatted_prefs = str(df_p) if df_p else "None"

    result = chain.run(tasks=formatted_tasks, preferences=formatted_prefs)

  
    return json.loads(result)
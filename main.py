# main.py — Schedule Optimizer Multi-Agent System
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Import agents
from agent1 import plan_day
from agent2 import process_locations, GeolocationResult, validate_locations
from agent3 import run_reasoner_agent, ScheduleOptimizationRequest

# Load environment variables
load_dotenv()

# Define data models
class Location(BaseModel):
    id: str
    name: str
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    is_fixed: bool = True

class Task(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    duration_minutes: int
    location_id: str  # Reference to location ID
    priority: str = "medium"  # low, medium, high
    constraints: Dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None

class ScheduleItem(BaseModel):
    task_id: str
    start_time: datetime
    end_time: datetime
    location_id: str

class SystemState(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]
    tasks: List[Task]
    locations: List[Location]
    schedule: List[ScheduleItem]
    travel_info: List[Dict[str, Any]]
    current_agent: str
    agent_messages: Dict[str, List[str]]
    optimization_result: Optional[Dict[str, Any]]
    iteration_count: int
    max_iterations: int = 3

# Initialize the graph
workflow = StateGraph(SystemState)

# Agent 1: Planner
def planner_agent(state: SystemState) -> SystemState:
    """Agent 1: Plans and categorizes tasks"""
    print("\n=== Planner Agent (Agent 1) ===")
    
    try:
        # Extract tasks and user preferences
        tasks = state.get("tasks", [])
        user_preferences = state.get("user_preferences", {})
        
        # Convert tasks to the format expected by plan_day
        task_input = [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description or "",
                "duration_minutes": t.duration_minutes,
                "priority": t.priority,
                "constraints": t.constraints,
                "category": t.category
            }
            for t in tasks
        ]
        
        # Call the planner with error handling
        try:
            planned_tasks = plan_day(task_input, df_d={}, df_p=user_preferences)
            
            # Ensure we have a list of tasks
            if not isinstance(planned_tasks, list):
                if isinstance(planned_tasks, dict) and 'tasks' in planned_tasks:
                    planned_tasks = planned_tasks['tasks']
                elif isinstance(planned_tasks, str):
                    # Try to parse as JSON
                    try:
                        import json
                        parsed = json.loads(planned_tasks)
                        if isinstance(parsed, dict) and 'tasks' in parsed:
                            planned_tasks = parsed['tasks']
                        elif isinstance(parsed, list):
                            planned_tasks = parsed
                    except json.JSONDecodeError:
                        print("Warning: Could not parse planner response as JSON")
                        planned_tasks = task_input
                
                # If still not a list, use the original tasks
                if not isinstance(planned_tasks, list):
                    print("Warning: Planner did not return a valid task list, using original tasks")
                    planned_tasks = task_input
                    
        except Exception as e:
            print(f"Error in planner: {str(e)}")
            planned_tasks = task_input
        
        # Update tasks with planning results
        updated_tasks = []
        for task in tasks:
            planned_task = next((pt for pt in planned_tasks if isinstance(pt, dict) and pt.get("id") == task.id), None)
            if planned_task:
                # Update task with planning results
                updated_task = task.model_dump()
                updated_task.update({
                    "priority": planned_task.get("priority", task.priority),
                    "category": planned_task.get("category", task.category),
                    "constraints": {**task.constraints, **planned_task.get("constraints", {})}
                })
                updated_tasks.append(Task(**updated_task))
            else:
                updated_tasks.append(task)
        
        return {
            **state,
            "tasks": updated_tasks,
            "current_agent": "geolocator",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "planner": ["Tasks have been planned and prioritized"]
            }
        }
        
    except Exception as e:
        error_msg = f"Planner Agent Error: {str(e)}"
        print(error_msg)
        return {
            **state,
            "current_agent": "geolocator",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "planner": [error_msg]
            }
        }

# Agent 2: Geolocator
def geolocator_agent(state: SystemState) -> SystemState:
    """Agent 2: Handles geolocation and travel information"""
    print("\n=== Geolocator Agent (Agent 2) ===")
    
    try:
        tasks = state.get("tasks", [])
        locations = state.get("locations", [])
        
        # Process locations with geocoding
        location_dicts = [loc.model_dump() for loc in locations]
        result = process_locations(location_dicts)
        
        # Update locations with geocoded data
        updated_locations = []
        for loc in locations:
            updated_loc = next(
                (l for l in result.locations if l["id"] == loc.id),
                loc.dict()
            )
            updated_locations.append(Location(**updated_loc))
        
        # Create a mapping of location IDs to task IDs
        location_to_task = {}
        for task in tasks:
            if hasattr(task, 'location_id'):
                location_to_task[task.location_id] = task.id
        
        # Prepare travel matrix in the format expected by the system
        travel_info = []
        for t in result.travel_matrix:
            from_loc = t["from_location_id"]
            to_loc = t["to_location_id"]
            
            # Get corresponding task IDs or use location IDs as fallback
            from_task = location_to_task.get(from_loc, f"task_{from_loc}")
            to_task = location_to_task.get(to_loc, f"task_{to_loc}")
            
            travel_info.append({
                "from_location_id": from_loc,
                "to_location_id": to_loc,
                "from_task_id": from_task,
                "to_task_id": to_task,
                "distance_km": t.get("distance_km", 10),  # Default to 10 km if not provided
                "travel_time_minutes": t.get("travel_time_min", 15),  # Default to 15 min if not provided
                "travel_time_min": t.get("travel_time_min", 15)  # For backward compatibility
            })
        
        # Validate locations
        validation = validate_locations([loc.model_dump() for loc in updated_locations])
        
        messages = ["Geolocation processing complete"]
        if not validation["is_valid"]:
            messages.append(f"Warning: {validation['message']}")
        
        return {
            **state,
            "locations": updated_locations,
            "travel_info": travel_info,
            "current_agent": "reasoner",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "geolocator": messages
            },
            "iteration_count": state.get("iteration_count", 0) + 1
        }
        
    except Exception as e:
        error_msg = f"Geolocator Agent Error: {str(e)}"
        print(error_msg)
        return {
            **state,
            "current_agent": "reasoner",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "geolocator": [error_msg]
            },
            "iteration_count": state.get("iteration_count", 0) + 1
        }

# Agent 3: Reasoner
def reasoner_agent(state: SystemState) -> SystemState:
    """Agent 3: Optimizes the schedule based on constraints"""
    print("\n=== Reasoner Agent (Agent 3) ===")
    
    try:
        # Prepare data for optimization
        tasks = state.get("tasks", [])
        locations = state.get("locations", [])
        travel_info = state.get("travel_info", [])
        
        # Create a mapping of location IDs to location dictionaries
        location_map = {loc.id: loc.model_dump() for loc in locations}
        
        # Prepare tasks with location information
        tasks_with_locations = []
        for task in tasks:
            task_dict = task.model_dump()
            task_location = location_map.get(task.location_id, {})
            task_dict['location'] = task_location
            
            # Ensure all required fields are present
            task_dict.setdefault('start_time', '09:00')  # Default start time
            task_dict.setdefault('end_time', '10:00')    # Default end time
            task_dict.setdefault('name', f'Task {task.id}')  # Default name
            
            tasks_with_locations.append(task_dict)
            
        # Add debug prints
        print("\n=== Debug: Tasks with Locations ===")
        for task in tasks_with_locations:
            print(f"Task: {task.get('name')} (ID: {task.get('id')}) at location: {task.get('location', {}).get('id', 'unknown')}")
        
        print("\n=== Debug: Locations ===")
        for loc in locations:
            loc_dict = loc.model_dump()
            print(f"Location: {loc_dict.get('id')} - {loc_dict.get('name')}")
        
        print("\n=== Debug: Travel Info ===")
        for trip in travel_info:
            print(f"From {trip.get('from_task_id')} to {trip.get('to_task_id')}: {trip.get('travel_time_min', '?')} min")
        
        # Prepare the state for the reasoner agent
        reasoner_state = {
            'tasks': tasks_with_locations,
            'locations': [loc.model_dump() for loc in locations],
            'travel_info': travel_info,  # This is the travel info from the geolocator
            'constraints': state.get('constraints', {
                'max_daily_hours': 8,
                'min_break_between_tasks': 15  # minutes
            }),
            'preferences': state.get('user_preferences', {
                'preferred_start_time': '09:00',
                'preferred_end_time': '17:00',
                'lunch_break': '12:00-13:00'
            }),
            'agent_messages': state.get('agent_messages', {})
        }
        
        # Run the optimization
        print("\n=== Debug: Running reasoner agent ===")
        print(f"Task count: {len(tasks_with_locations)}")
        print(f"Location count: {len(locations)}")
        print(f"Travel info count: {len(travel_info)}")
        
        optimization_result = run_reasoner_agent(reasoner_state)
        
        print(f"\n=== Debug: Optimization Result ===")
        print(f"Type: {type(optimization_result)}")
        print(f"Keys: {optimization_result.keys() if isinstance(optimization_result, dict) else 'N/A'}")
        print(f"Content: {json.dumps(optimization_result, indent=2) if isinstance(optimization_result, dict) else optimization_result}")
        
        # Update the schedule
        schedule = []
        if isinstance(optimization_result, dict):
            # Check if we have an 'optimized_schedule' key first, then fall back to 'schedule'
            schedule_data = optimization_result.get('optimized_schedule', [])
            if not schedule_data:
                schedule_data = optimization_result.get('schedule', [])
                
            print(f"\n=== Debug: Schedule Data ===")
            print(f"Type: {type(schedule_data)}")
            print(f"Length: {len(schedule_data) if isinstance(schedule_data, (list, dict)) else 'N/A'}")
            if schedule_data and isinstance(schedule_data, list) and len(schedule_data) > 0:
                print(f"First item type: {type(schedule_data[0])}")
                print(f"First item keys: {schedule_data[0].keys() if isinstance(schedule_data[0], dict) else 'N/A'}")
                
            for idx, item in enumerate(schedule_data):
                try:
                    if not isinstance(item, dict):
                        print(f"Warning: Item at index {idx} is not a dictionary: {item}")
                        continue
                        
                    # Get required fields with defaults
                    task_id = item.get('task_id', f'task_{idx + 1}')
                    location_id = item.get('location_id', 'unknown')
                    
                    # Handle start time with validation
                    start_time = item.get('start_time', '09:00')
                    if isinstance(start_time, str):
                        try:
                            start_time = datetime.strptime(start_time, '%H:%M')
                        except ValueError:
                            print(f"Warning: Invalid start_time format '{start_time}'. Using default 09:00")
                            start_time = datetime.strptime('09:00', '%H:%M')
                    
                    # Handle end time with validation
                    end_time = item.get('end_time', '10:00')
                    if isinstance(end_time, str):
                        try:
                            end_time = datetime.strptime(end_time, '%H:%M')
                        except ValueError:
                            print(f"Warning: Invalid end_time format '{end_time}'. Using default 10:00")
                            end_time = datetime.strptime('10:00', '%H:%M')
                    
                    # Ensure end time is after start time
                    if end_time <= start_time:
                        print(f"Warning: End time {end_time} is before or equal to start time {start_time} for task {task_id}. Adjusting...")
                        end_time = datetime(
                            start_time.year, start_time.month, start_time.day,
                            start_time.hour, min(start_time.minute + 60, 59)  # Default to 1 hour duration
                        )
                    
                    schedule_item = ScheduleItem(
                        task_id=task_id,
                        start_time=start_time,
                        end_time=end_time,
                        location_id=location_id
                    )
                    schedule.append(schedule_item)
                    
                    print(f"Added schedule item: Task {task_id} at {location_id} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
                    
                except Exception as e:
                    print(f"Error processing schedule item at index {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"Problematic item data: {item}")
        else:
            print(f"Warning: Unexpected optimization result type: {type(optimization_result)}")
        
        # Check if we should continue iterating
        iteration_count = state.get('iteration_count', 0) + 1
        max_iterations = state.get('max_iterations', 3)
        
        if iteration_count >= max_iterations or optimization_result.get('converged', True):
            next_agent = END
            messages = ["Optimization complete"]
        else:
            next_agent = "planner"
            messages = ["Continuing optimization"]
        
        # Ensure we have a valid next_agent
        if next_agent is None:
            next_agent = END
            messages.append("No next agent specified, ending optimization")
        
        # Prepare the next state
        next_state = {
            **state,
            "schedule": schedule,
            "current_agent": next_agent,
            "agent_messages": {
                **state.get("agent_messages", {}),
                "reasoner": messages
            },
            "iteration_count": state.get("iteration_count", 0) + 1
        }
        
        # Return the next state with the appropriate transition
        if next_agent == END:
            return {
                **next_state,
                "__end__": True  # Signal the end of the workflow
            }
        return next_state
        
    except Exception as e:
        error_msg = f"Reasoner Agent Error: {str(e)}"
        print(error_msg)
        
        # Create a default schedule if optimization failed
        schedule = []
        tasks = state.get('tasks', [])
        
        # Ensure we have a list of tasks
        if not isinstance(tasks, list):
            tasks = []
            
        for i, task in enumerate(tasks):
            try:
                # Create a simple default schedule
                start_hour = 9 + i  # Space tasks out by 1 hour
                task_id = getattr(task, 'id', f'task_{i+1}')
                location_id = getattr(task, 'location_id', f'loc_{(i % 3) + 1}')  # Cycle through locations
                
                schedule_item = ScheduleItem(
                    task_id=task_id,
                    start_time=datetime.strptime(f"{start_hour}:00", "%H:%M"),
                    end_time=datetime.strptime(f"{start_hour + 1}:00", "%H:%M"),
                    location_id=location_id
                )
                schedule.append(schedule_item)
            except Exception as task_error:
                print(f"Warning: Could not create default schedule item: {task_error}")
        
        # Create the next state
        next_state = {
            **state,
            "schedule": schedule,
            "current_agent": END,
            "agent_messages": {
                **state.get("agent_messages", {}),
                "reasoner": [f"Optimization completed with errors: {str(e)}. Using default schedule."]
            },
            "iteration_count": state.get("iteration_count", 0) + 1
        }
        
        # Return the next state
        return next_state

# Add nodes to the workflow
workflow.add_node("planner", planner_agent)
workflow.add_node("geolocator", geolocator_agent)
workflow.add_node("reasoner", reasoner_agent)

# Define the graph edges
workflow.add_edge(START, "planner")  # Start with the planner
workflow.add_edge("planner", "geolocator")
workflow.add_edge("geolocator", "reasoner")
workflow.add_edge("reasoner", END)

# Define conditional edges for the planner
workflow.add_conditional_edges(
    "planner",
    lambda state: state.get("current_agent", "planner"),
    {
        "geolocator": "geolocator",
        "reasoner": "reasoner",
        "done": END,
        END: END
    }
)

# Define conditional edges for the geolocator
workflow.add_conditional_edges(
    "geolocator",
    lambda state: state.get("current_agent", "geolocator"),
    {
        "reasoner": "reasoner",
        "planner": "planner",
        "done": END,
        END: END
    }
)

# Define conditional edges for the reasoner
workflow.add_conditional_edges(
    "reasoner",
    lambda state: state.get("current_agent", "reasoner"),
    {
        "planner": "planner",
        "geolocator": "geolocator",
        "done": END,
        END: END
    }
)

# Compile the workflow
app = workflow.compile()

def run_schedule_optimizer(
    tasks: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    user_preferences: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run the schedule optimization workflow.
    
    Args:
        tasks: List of task dictionaries
        locations: List of location dictionaries
        user_preferences: Dictionary of user preferences
        constraints: Dictionary of scheduling constraints
        max_iterations: Maximum number of optimization iterations
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing the final state of the workflow
    """
    try:
        print("Starting schedule optimization...\n")
        
        # Convert input dictionaries to Pydantic models
        task_models = [Task(**task) for task in tasks]
        location_models = [Location(**loc) for loc in locations]
        
        # Initialize state
        initial_state = {
            "tasks": task_models,
            "locations": location_models,
            "schedule": [],
            "travel_info": [],
            "current_agent": "planner",
            "agent_messages": {},
            "optimization_result": None,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "user_preferences": user_preferences or {},
            "constraints": constraints or {},
            "messages": []
        }
        
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Ensure we have a valid schedule
        if not final_state.get("schedule"):
            print("Warning: No schedule was generated. Creating a default schedule...")
            schedule = []
            for i, task in enumerate(final_state.get("tasks", [])):
                start_hour = 9 + i  # Space tasks out by 1 hour
                schedule_item = ScheduleItem(
                    task_id=getattr(task, 'id', f'task_{i+1}'),
                    start_time=datetime.strptime(f"{start_hour}:00", "%H:%M"),
                    end_time=datetime.strptime(f"{start_hour + 1}:00", "%H:%M"),
                    location_id=getattr(task, 'location_id', f'loc_{(i % 3) + 1}')
                )
                schedule.append(schedule_item)
            final_state["schedule"] = schedule
        
        # Convert the final state to a dictionary
        result = {
            "schedule": [item.model_dump() for item in final_state.get("schedule", [])],
            "tasks": [task.model_dump() for task in final_state.get("tasks", [])],
            "locations": [loc.model_dump() for loc in final_state.get("locations", [])],
            "optimization_result": final_state.get("optimization_result"),
            "agent_messages": final_state.get("agent_messages", {}),
            "iterations": final_state.get("iteration_count", 0)
        }
        
        # Print summary
        print("\n=== Optimization Complete ===")
        print(f"Completed in {result['iterations']} iterations")
        
        print("\n=== Schedule ===")
        for item in result["schedule"]:
            task = next((t for t in result["tasks"] if t["id"] == item["task_id"]), {"name": "Unknown Task"})
            location = next((loc["name"] for loc in result["locations"] if loc["id"] == item["location_id"]), "Unknown Location")
            print(f"- {item['start_time']} - {item['end_time']}: {task.get('name')} at {location}")
        
        print("\n=== Agent Messages ===")
        for agent, messages in result["agent_messages"].items():
            print(f"{agent.capitalize()}:")
            for msg in messages:
                print(f"  - {msg}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error during optimization: {str(e)}"
        print(error_msg)
        
        # Create a default result with error information
        return {
            "schedule": [],
            "tasks": [],
            "locations": [],
            "optimization_result": {"error": str(e)},
            "agent_messages": {"error": [error_msg]},
            "iterations": 0
        }

# Example usage
if __name__ == "__main__":
    # Example data with real coordinates for New York City
    example_locations = [
        {
            "id": "loc1",
            "name": "Home",
            "address": "350 5th Ave, New York, NY 10118",
            "lat": 40.7484,
            "lng": -73.9857,
            "is_fixed": True
        },
        {
            "id": "loc2",
            "name": "Work",
            "address": "767 5th Ave, New York, NY 10153",
            "lat": 40.7638,
            "lng": -73.9730,
            "is_fixed": True
        },
        {
            "id": "loc3",
            "name": "Grocery Store",
            "address": "11 Madison Ave, New York, NY 10010",
            "lat": 40.7416,
            "lng": -73.9872,
            "is_fixed": False
        }
    ]
    
    example_tasks = [
        {
            "id": "task1",
            "name": "Morning Meeting",
            "description": "Team sync meeting",
            "duration_minutes": 60,
            "location_id": "loc2",
            "priority": "high",
            "constraints": {
                "time_windows": [{"start": "09:00", "end": "17:00"}],
                "required_days": ["monday", "wednesday", "friday"]
            },
            "category": "work"
        },
        {
            "id": "task2",
            "name": "Grocery Shopping",
            "description": "Weekly grocery shopping",
            "duration_minutes": 90,
            "location_id": "loc3",
            "priority": "medium",
            "constraints": {
                "time_windows": [{"start": "10:00", "end": "20:00"}],
                "frequency": "weekly"
            },
            "category": "shopping"
        },
        {
            "id": "task3",
            "name": "Exercise",
            "description": "Daily workout",
            "duration_minutes": 45,
            "location_id": "loc1",
            "priority": "high",
            "constraints": {
                "time_windows": [
                    {"start": "06:00", "end": "08:00"},
                    {"start": "18:00", "end": "20:00"}
                ]
            },
            "category": "health"
        }
    ]
    
    # User preferences
    user_preferences = {
        "work_hours": {"start": "09:00", "end": "17:00"},
        "preferred_work_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "preferred_categories_order": ["health", "work", "shopping", "personal"],
        "travel_mode": "driving",
        "max_travel_time_minutes": 30
    }
    
    # Constraints
    constraints = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
        "max_hours_per_day": 10,
        "min_break_between_tasks_minutes": 15
    }
    
    try:
        # Run the optimizer
        print("Starting schedule optimization...")
        result = run_schedule_optimizer(
            tasks=example_tasks,
            locations=example_locations,
            user_preferences=user_preferences,
            constraints=constraints,
            max_iterations=3
        )
        
        # Generate output strings
        output_lines = []
        output_lines.append("=== Schedule ===")
        
        for item in result["schedule"]:
            task = next((t for t in result["tasks"] if t["id"] == item["task_id"]), None)
            if not task:
                output_lines.append(f"- {item['start_time']} - {item['end_time']}: Task {item['task_id']} (location not found)")
                continue
            
            # Get the location from the task's location_id
            location = next((l for l in result["locations"] if l["id"] == task.get("location_id")), None)
            location_name = location["name"] if location else "Unknown Location"
            output_lines.append(f"- {item['start_time']} - {item['end_time']}: {task['name']} at {location_name}")
        
        output_lines.append("")
        output_lines.append("=== Agent Messages ===")
        for agent, messages in result["agent_messages"].items():
            output_lines.append(f"{agent.capitalize()}:")
            for msg in messages:
                output_lines.append(f"  - {msg}")
        
        # Write to file
        output_path = "schedule_results.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        # Print to console
        print("\n=== Optimization Complete ===")
        print(f"Results saved to {output_path}")
        print('\n'.join(output_lines))
                
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise

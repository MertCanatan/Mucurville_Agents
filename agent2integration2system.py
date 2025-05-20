from agent1 import plan_day
# this file integrate the standalone agent2 into the entire agentic system that incorporates agents 1 and 3 as well.
from typing import Annotated, Dict, List, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.distance import distance
import os
import json
import requests

# This is a simplified integration example showing how Agent 2 fits into the multi-agent system
# Initialize LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest",api_key = "PROVIDED_API_KEY")

# Define common models
class UserPreferences(BaseModel):
    preferred_transit_modes: List[str] = Field(default_factory=list)
    preferred_times: Dict[str, Any] = Field(default_factory=dict)
    max_travel_time: Optional[int] = None

class Location(BaseModel):
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
    location: Optional[Location] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "medium"  # low, medium, high

class ScheduleItem(BaseModel):
    task_id: str
    start_time: datetime
    end_time: datetime
    location: Location

class TravelInfo(BaseModel):
    from_task_id: str
    to_task_id: str
    travel_time_minutes: int
    distance_km: float
    transit_mode: str

# State definitions for the multi-agent system
class SystemState(TypedDict):
    messages: Annotated[list, add_messages]
    user_data: Dict
    tasks: List[Task]
    locations: List[Location]
    schedule: List[ScheduleItem]
    travel_info: List[TravelInfo]
    current_agent: str
    agent_messages: Dict[str, List]

# Agent 1: Planner Functions
def planner_agent(state: SystemState) -> SystemState:
    """Agent 1 plans the tasks and categorizes them"""
    print("Agent 1 (Planner) is processing...")

    try:
        raw_tasks = state.get("tasks", [])
        user_data = state.get("user_data", {})
        preferences = user_data.get("preferences", {})
        device_data = state.get("device_data", {}) if "device_data" in state else {}

        # Run the planner logic
        prioritized_tasks = plan_day(raw_tasks, df_d=device_data, df_p=preferences)

        state["tasks"] = prioritized_tasks

        return {
            **state,
            "current_agent": "geolocator",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "planner": ["Tasks have been planned, prioritized, and categorized"]
            }
        }
    except Exception as e:
        print(f"Planner Agent Error: {e}")
        return {
            **state,
            "current_agent": "geolocator",
            "agent_messages": {
                **state.get("agent_messages", {}),
                "planner": [f"Error during planning: {e}"]
            }
        }

# Agent 2: Geolocator Functions
def geolocator_agent(state: SystemState) -> SystemState:
    """Agent 2 handles geolocation and travel planning"""
    print("Agent 2 (Geolocator) is processing...")
    
    # Initialize geolocator
    geolocator = Nominatim(user_agent="schedule_optimizer_demo")

    tasks = state.get("tasks", [])
    locations = state.get("locations", [])
    
    # Simulate geocoding and location processing
    # geocoded_locations = []
    # for location in locations:
    #     if location.is_fixed and not (location.lat and location.lng):
    #         # In a real implementation, this would call a geocoding service
    #         print(f"Geocoding location: {location.name}")
    #         # Simulate coordinates for demonstration
    #         location.lat = 37.7749  # Example latitude
    #         location.lng = -122.4194  # Example longitude
    #     geocoded_locations.append(location)
    
    # Use real geopy for geocoding fixed locations with addresses
    updated_locations = []
    for location in locations:
        if location.is_fixed and location.address and (not location.lat or not location.lng):
            try:
                print(f"Geocoding address: {location.address}")
                geocode_result = geolocator.geocode(location.address)
                if geocode_result:
                    location.lat = geocode_result.latitude
                    location.lng = geocode_result.longitude
                    print(f"Geocoded {location.name} to {location.lat}, {location.lng}")
                else:
                    print(f"Could not geocode address: {location.address}")
            except Exception as e:
                print(f"Error geocoding {location.address}: {e}")
        updated_locations.append(location)

    # Create a sample travel info
    # travel_info = []
    # if len(tasks) >= 2:
    #     travel_info.append(
    #         TravelInfo(
    #             from_task_id=tasks[0].id,
    #             to_task_id=tasks[1].id, 
    #             travel_time_minutes=15,
    #             distance_km=5.2,
    #             transit_mode="driving"
    #         )
    #     )
    
    # Calculate travel times between consecutive tasks using geocoded locations
    travel_info = []
    # Ensure tasks are ordered for travel time calculation (assuming order matters)
    ordered_tasks = sorted(tasks, key=lambda t: t.id) # Assuming task id implies order, adjust if needed

    for i in range(len(ordered_tasks) - 1):
        from_task = ordered_tasks[i]
        to_task = ordered_tasks[i + 1]

        from_location = next((loc for loc in updated_locations if loc.name == from_task.location.name), None) # Assuming location name is unique enough
        to_location = next((loc for loc in updated_locations if loc.name == to_task.location.name), None)
        
        if from_location and to_location and from_location.lat and from_location.lng and to_location.lat and to_location.lng:
            try:
                dist = distance(
                    (from_location.lat, from_location.lng),
                    (to_location.lat, to_location.lng)
                ).kilometers
                
                # Estimate travel time (very rough estimate - 30 km/h average speed)
                travel_time_minutes = int(dist * 2) # ~30 km/h

                travel_info.append(TravelInfo(
                    from_task_id=from_task.id,
                    to_task_id=to_task.id,
                    travel_time_minutes=travel_time_minutes,
                    distance_km=dist,
                    transit_mode="driving" # Assuming driving for now
                ))
                print(f"Calculated travel from {from_task.name} to {to_task.name}: {travel_time_minutes} minutes ({dist:.2f} km)")

            except Exception as e:
                 print(f"Error calculating travel time between {from_task.name} and {to_task.name}: {e}")


    # Create a simple draft schedule (keep existing logic but use updated locations)
    current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    schedule = []
    
    # Use ordered tasks for scheduling as well
    for i, task in enumerate(ordered_tasks):
        # Find the updated location for this task
        task_location = next((loc for loc in updated_locations if loc.name == task.location.name), None)

        schedule_item = ScheduleItem(
            task_id=task.id,
            start_time=current_time,
            end_time=current_time + timedelta(minutes=task.duration_minutes),
            location=task_location or Location(name="Unknown", is_fixed=False) # Use updated location
        )
        schedule.append(schedule_item)
        
        # Update the time for the next task, including travel time
        current_time = schedule_item.end_time
        if i < len(ordered_tasks) - 1:
            next_task = ordered_tasks[i+1]
            travel = next((t for t in travel_info if t.from_task_id == task.id and t.to_task_id == next_task.id), None)
            if travel:
                current_time += timedelta(minutes=travel.travel_time_minutes)

    
    return {
        "locations": updated_locations, # Return updated locations with coordinates
        "travel_info": travel_info,
        "schedule": schedule,
        "current_agent": "reasoner",
        "agent_messages": {
            **state.get("agent_messages", {}),
            "geolocator": ["Locations geocoded and travel times calculated"]
        }
    }

# Agent 3: Reasoner Functions
def reasoner_agent(state: SystemState) -> SystemState:
    """Agent 3 optimizes the schedule based on constraints"""
    print("Agent 3 (Reasoner) is processing...")
    
    # This would have the logic for Agent 3
    # For demonstration purposes, we'll just pass through

    get_schedule = state.get("schedule", [])
    travel_info = state.get("travel_info", [])
    locations = state.get("locations", [])
    user_data = state.get("user_data", {})
    preferences = user_data.get("preferences", {})

    print("Current Schedule:")
    for item in get_schedule:   
        start_time = item.start_time.strftime("%H:%M")
        end_time = item.end_time.strftime("%H:%M")
        task_id = item.task_id
        location = item.location.name if item.location else "Unknown"
        
        print(f"{start_time} - {end_time}: Task {task_id} at {location}")
    print("Travel Info:")
    for info in travel_info:
        print(f"From {info.from_task_id} to {info.to_task_id}: {info.travel_time_minutes} minutes, {info.distance_km:.2f} km")
    print("Locations:")
    for loc in locations:
        print(f"{loc.name}: {loc.lat}, {loc.lng}")
    print("User Preferences:")
    print(preferences)

    llm_response_string = llm.invoke(
        f'''You are the best reasoning agent. You will be presented a schedule and list of preferences of a person. Some of the tasks have fixed timelines. Your job is to judge whether the schedule is feasible based on the travel times between tasks and the locations of the tasks. Also consider the user preferences. The tasks are as follows:\n{get_schedule}\nThe travel times are as follows:\n{travel_info}\nThe locations are as follows:\n{locations}\nThe user preferences are as follows\n{preferences}. If the schedule is feasible, return 'yes'. If not, return 'no' and suggest an optimized schedule.
        e.g. {{
            'is_feasible': 'yes',
            'schedule': {{
                'task_id': 'task1',
                'start_time': '2025-05-20 09:00:00',
                'end_time': '2025-05-20 09:30:00',
                'location': 'Grocery Store'
            }},
            'travel_info': {{
                'from_task_id': 'task1',            
                'to_task_id': 'task2',
                'travel_time_minutes': 16,
                'distance_km': 8.29

            }},
            'locations': {{
                'name': 'Grocery Store',
                'address': 'Ferry Building, San Francisco, CA'
            }},
            'user_preferences': {{
                'preferred_transit_modes': ['driving', 'walking'],
                'max_travel_time': 30,
                }}      
        }}''')
         # Simulate LLM response 


    #llm_response_string = """content='{\n    "feasible": "yes",\n    "schedule": [\n        {\n            "task_id": "task1",\n            "start_time": "2025-05-20 09:00:00",\n            "end_time": "2025-05-20 09:30:00",\n            "location": "Grocery Store"\n        },\n        {\n            "task_id": "task2",\n            "start_time": "2025-05-20 09:46:00",\n            "end_time": "2025-05-20 10:46:00",\n            "location": "Medical Center"\n        }\n    ],\n    "travel_info": [\n        {\n            "from_task_id": "task1",\n            "to_task_id": "task2",\n            "travel_time_minutes": 16,\n            "distance_km": 8.29,\n            "transit_mode": "driving"\n        }\n    ],\n    "locations": [\n        {\n            "name": "Grocery Store",\n            "address": "Ferry Building, San Francisco, CA"\n        },\n        {\n            "name": "Medical Center",\n            "address": "Golden Gate Park, San Francisco, CA"\n        }\n    ]\n}' additional_kwargs={} response_metadata={'id': 'msg_01M2PXzuxWgEbwnXegtuhabF', 'model': 'claude-3-5-sonnet-latest', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 543, 'output_tokens': 307, 'server_tool_use': None}, 'model_name': 'claude-3-5-sonnet-latest'} id='run--1566431c-e3a3-41f0-a567-f659a2b09801-0' usage_metadata={'input_tokens': 543, 'output_tokens': 307, 'total_tokens': 850, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}"""

    # 1. Extract the JSON string
    # This assumes the format is always content='{...}'
    # We need to find the first '{' and the last '}' and slice the string.
    # A more robust way might be using regex if the 'content=' prefix is variable.
    try:
        json_start = str(llm_response_string).find('{')
        json_end = str(llm_response_string).find('}') + 1
        json_content_string = str(llm_response_string)[json_start:json_end]
        print(f"Extracted JSON string: {json_content_string}")
        json_content_string = json_content_string.replace(' ', '') # Replace non-breaking space with regular space

        # 2. Parse the JSON string into a Python dictionary
        parsed_data = json.loads(json_content_string)
        # 3. Now you can access the data using dictionary keys
        print(
            f"Parsed JSON: {json_content_string}"
        )

        print(
            f"Parsed data: {parsed_data}"
        )

        # Now you can access the data using dictionary keys
        feasible = parsed_data.get('feasible')
        schedule = parsed_data.get('schedule')
        travel_info = parsed_data.get('travel_info')
        locations = parsed_data.get('locations')

        print(f"Feasible: {feasible}")
        print("\nSchedule:")
        for task in schedule:
            print(f"  Task ID: {task.get('task_id')}")
            print(f"  Start Time: {task.get('start_time')}")
            print(f"  End Time: {task.get('end_time')}")
            print(f"  Location: {task.get('location')}")
            print("-" * 20)

        print("\nTravel Info:")
        for travel in travel_info:
            print(f"  From Task ID: {travel.get('from_task_id')}")
            print(f"  To Task ID: {travel.get('to_task_id')}")
            print(f"  Travel Time (minutes): {travel.get('travel_time_minutes')}")
            print(f"  Distance (km): {travel.get('distance_km')}")
            print(f"  Transit Mode: {travel.get('transit_mode')}")
            print("-" * 20)

        print("\nLocations:")
        for loc in locations:
            print(f"  Name: {loc.get('name')}")
            print(f"  Address: {loc.get('address')}")
            print("-" * 20)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic string portion: {json_content_string}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    #print(f"LLM Response: {llm_response_string}")


    # For demonstration, we will just return the state as is
    # Note: The LLM call is commented out for demonstration purposes
    # In a real implementation, this would call the LLM to generate an optimized schedule
    # For now, we'll just return the state as is
    # Note: The LLM call is commented out for demonstration purposes
    #response = llm.generate(messages=[{"role": "user", "content": "Optimize the schedule"}])


    # Here you would implement the reasoning logic to optimize the schedule
    
    return {
        "current_agent": "done",
        "agent_messages": {
            **state.get("agent_messages", {}),
            "reasoner": ["Schedule has been optimized"]
        },
        "messages": [
            {
                "role": "assistant", 
                "content": "I've processed your schedule and optimized it based on locations and travel times."
            }
        ]
    }

# Define agent router
def route_to_next_agent(state: SystemState) -> Dict[str, Any]:
    """Route to the appropriate agent based on current_agent field"""
    current_agent = state.get("current_agent", "planner")
    
    if current_agent == "planner":
        return {"next": "planner_agent"}
    elif current_agent == "geolocator":
        return {"next": "geolocator_agent"}
    elif current_agent == "reasoner":
        return {"next": "reasoner_agent"}
    else:
        return {"next": "end"}

# Build the graph
graph_builder = StateGraph(SystemState)

# Add nodes
graph_builder.add_node("router", route_to_next_agent)
graph_builder.add_node("planner_agent", planner_agent)
graph_builder.add_node("geolocator_agent", geolocator_agent)
graph_builder.add_node("reasoner_agent", reasoner_agent)

# Add edges
graph_builder.add_edge(START, "router")

# Add conditional edges from router
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next", "end"),
    {
        "planner_agent": "planner_agent",
        "geolocator_agent": "geolocator_agent",
        "reasoner_agent": "reasoner_agent",
        "end": END
    }
)

# Connect agents back to router
graph_builder.add_edge("planner_agent", "router")
graph_builder.add_edge("geolocator_agent", "router")
graph_builder.add_edge("reasoner_agent", "router")

# Compile the graph
graph = graph_builder.compile()

# Example data for testing
example_tasks = [
    Task(
        id="task1",
        name="Buy groceries",
        description="Get items for dinner",
        duration_minutes=30,
        location=Location(
            name="Grocery Store",
            address="Ferry Building, San Francisco, CA", # Updated address for demo
            is_fixed=True # Changed to fixed for geocoding demo
        ),
        priority="high"
    ),
    Task(
        id="task2",
        name="Doctor appointment",
        description="Annual checkup",
        duration_minutes=60,
        location=Location(
            name="Medical Center",
            address="Golden Gate Park, San Francisco, CA", # Updated address for demo
            is_fixed=True # Changed to fixed for geocoding demo
        ),
        constraints={"fixed_time": "2023-05-18T14:00:00"},
        priority="high"
    ),
]

# Example function to run the multi-agent system
def run_scheduler_system(tasks=None):
    if tasks is None:
        tasks = example_tasks
    
    locations = [task.location for task in tasks if task.location]
    
    initial_state = {
        "messages": [{"role": "user", "content": "Please optimize my schedule for today."}],
        "user_data": {
            "preferences": {
                "preferred_transit_modes": ["driving", "walking"],
                "max_travel_time": 30
            }
        },
        "tasks": tasks,
        "locations": locations,
        "schedule": [],
        "travel_info": [],
        "current_agent": "planner",
        "agent_messages": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\nFinal Messages:")
    for message in result.get("messages", []):
        # Check the type of the message to determine the role
        if hasattr(message, 'type'): # Use .type attribute if available (common in LangChain)
            role = message.type
        elif message.__class__.__name__ == 'HumanMessage':
            role = 'user'
        elif message.__class__.__name__ == 'AIMessage':
            role = 'assistant'
        else:
            role = 'unknown'
        print(f"{role}: {message.content}")
    
    print("\nAgent Messages:")
    for agent, messages in result.get("agent_messages", {}).items():
        print(f"{agent}: {', '.join(messages)}")
    
    print("\nFinal Schedule:")
    for item in result.get("schedule", []):
        start_time = item.start_time.strftime("%H:%M")
        end_time = item.end_time.strftime("%H:%M")
        task_id = item.task_id
        location = item.location.name if item.location else "Unknown"
        
        print(f"{start_time} - {end_time}: Task {task_id} at {location}")
    
    return result

if __name__ == "__main__":
    run_scheduler_system()
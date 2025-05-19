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

# This is a simplified integration example showing how Agent 2 fits into the multi-agent system

# Load environment variables
load_dotenv()

# Initialize LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

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
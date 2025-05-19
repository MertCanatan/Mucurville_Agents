# agent 2 geolocator implementation standalone with dummy data (not actual maps api)
from typing import Annotated, Dict, List, Optional, TypedDict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json
from geopy.geocoders import Nominatim
from geopy.distance import distance
import requests
import polyline
import datetime
import time

# Load environment variables
load_dotenv()

# Initialize LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Define State type
class TaskLocation(BaseModel):
    task_id: str
    name: str
    is_fixed: bool
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class TaskWithTime(BaseModel):
    task_id: str
    name: str
    duration_minutes: int
    location: TaskLocation
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

class TravelInfo(BaseModel):
    from_task_id: str
    to_task_id: str
    travel_time_minutes: int
    distance_km: float

class GeolocatorState(TypedDict):
    messages: Annotated[list, add_messages]
    tasks: List[TaskWithTime]
    locations: List[TaskLocation]
    travel_info: List[TravelInfo]
    schedule_feasible: Optional[bool]
    schedule_feedback: Optional[str]

# Initialize geolocation services
geolocator = Nominatim(user_agent="schedule_optimizer")

def geocode_locations(state: GeolocatorState) -> GeolocatorState:
    """Geocode all locations in the tasks list"""
    locations = state.get("locations", [])
    updated_locations = []
    
    for location in locations:
        if not location.lat or not location.lng:
            try:
                # Only geocode if we have an address and don't already have coordinates
                if location.address:
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
    
    return {"locations": updated_locations}

def find_nearby_locations(state: GeolocatorState) -> GeolocatorState:
    """Find nearby locations for tasks with flexible locations"""
    tasks = state.get("tasks", [])
    locations = state.get("locations", [])
    
    # For demonstration, we're using a simplified approach
    # In a real implementation, you'd use Google Places API or similar
    
    updated_locations = locations.copy()
    
    for task in tasks:
        if not task.location.is_fixed and not (task.location.lat and task.location.lng):
            # For flexible locations, we would search for suitable nearby places
            # using Google Places API or similar
            # For now, we'll simulate this with a simple message
            print(f"Finding nearby locations for task: {task.name}")
            
            # This would be replaced with an actual API call
            # Example placeholder:
            # nearby_places = google_maps.find_nearby_places(
            #    type=task.category, 
            #    location=user_location, 
            #    radius=5000
            # )
            
            # For now, we'll just add a placeholder message
            feedback = f"Need to find a location for flexible task: {task.name}"
            if "schedule_feedback" in state and state["schedule_feedback"]:
                state["schedule_feedback"] += "\n" + feedback
            else:
                state["schedule_feedback"] = feedback
    
    return state

def calculate_travel_times(state: GeolocatorState) -> GeolocatorState:
    """Calculate travel times between locations"""
    tasks = state.get("tasks", [])
    travel_info = []
    
    # For demonstration purposes, we'll use a simple distance calculation
    # In a real implementation, you would use Google's Distance Matrix API or similar
    
    for i in range(len(tasks) - 1):
        from_task = tasks[i]
        to_task = tasks[i + 1]
        
        if (from_task.location.lat and from_task.location.lng and 
            to_task.location.lat and to_task.location.lng):
            
            # Calculate as-the-crow-flies distance
            dist = distance(
                (from_task.location.lat, from_task.location.lng),
                (to_task.location.lat, to_task.location.lng)
            ).kilometers
            
            # Estimate travel time (very rough estimate - 30 km/h average speed)
            # In a real implementation, use the Google Maps Direction API
            travel_time_minutes = int(dist * 2)  # Very rough estimate
            
            travel_info.append(TravelInfo(
                from_task_id=from_task.task_id,
                to_task_id=to_task.task_id,
                travel_time_minutes=travel_time_minutes,
                distance_km=dist
            ))
            
            print(f"Travel from {from_task.name} to {to_task.name}: {travel_time_minutes} minutes ({dist:.2f} km)")
    
    return {"travel_info": travel_info}

def create_schedule(state: GeolocatorState) -> GeolocatorState:
    """Create an initial schedule with time blocks"""
    tasks = state.get("tasks", [])
    travel_info = state.get("travel_info", [])
    
    # Start at 9 AM for this example
    current_time = datetime.datetime.now().replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    
    updated_tasks = []
    
    for i, task in enumerate(tasks):
        # Set start time for this task
        task_copy = task.model_copy()
        task_copy.start_time = current_time
        
        # Calculate end time based on duration
        current_time = current_time + datetime.timedelta(minutes=task.duration_minutes)
        task_copy.end_time = current_time
        
        updated_tasks.append(task_copy)
        
        # Add travel time to the next location if there is one
        if i < len(tasks) - 1:
            travel_time = 0
            for travel in travel_info:
                if travel.from_task_id == task.task_id and travel.to_task_id == tasks[i+1].task_id:
                    travel_time = travel.travel_time_minutes
                    break
            
            current_time = current_time + datetime.timedelta(minutes=travel_time)
    
    return {"tasks": updated_tasks}

def check_schedule_feasibility(state: GeolocatorState) -> GeolocatorState:
    """Check if the schedule is feasible"""
    tasks = state.get("tasks", [])
    
    # Example checks for feasibility:
    # 1. Check if the schedule extends too late
    end_time = tasks[-1].end_time if tasks and tasks[-1].end_time else None
    
    feedback = []
    schedule_feasible = True
    
    if end_time and end_time.hour >= 20:  # If schedule goes past 8 PM
        feedback.append(f"Schedule goes too late (ends at {end_time.strftime('%H:%M')})")
        schedule_feasible = False
    
    # 2. Check for locations without coordinates
    for task in tasks:
        if not (task.location.lat and task.location.lng):
            feedback.append(f"Task {task.name} doesn't have a geocoded location")
            schedule_feasible = False
    
    return {
        "schedule_feasible": schedule_feasible,
        "schedule_feedback": "\n".join(feedback) if feedback else "Schedule looks feasible"
    }

def send_to_reasoner(state: GeolocatorState) -> Dict[str, Any]:
    """Decide whether to send to Agent 3 (Reasoner) or need to rework"""
    schedule_feasible = state.get("schedule_feasible", False)
    
    if schedule_feasible:
        return {"next": "send_to_agent3"}
    else:
        return {"next": "rework_schedule"}

def send_to_agent3(state: GeolocatorState) -> GeolocatorState:
    """Mock function to send data to Agent 3"""
    feedback = "Schedule is feasible. Sending to Agent 3 (Reasoner) for optimization."
    
    # In a real implementation, this would format and send data to Agent 3
    
    return {
        "messages": [{"role": "assistant", "content": feedback}]
    }

def rework_schedule(state: GeolocatorState) -> GeolocatorState:
    """Rework the schedule to address feasibility issues"""
    feedback = f"Schedule needs reworking:\n{state.get('schedule_feedback', '')}"
    
    # In a real implementation, this would adjust the schedule
    # For now, we just acknowledge the issues
    
    return {
        "messages": [{"role": "assistant", "content": feedback}]
    }

# Create the graph
graph_builder = StateGraph(GeolocatorState)

# Add nodes
graph_builder.add_node("geocode_locations", geocode_locations)
graph_builder.add_node("find_nearby_locations", find_nearby_locations)
graph_builder.add_node("calculate_travel_times", calculate_travel_times)
graph_builder.add_node("create_schedule", create_schedule)
graph_builder.add_node("check_schedule_feasibility", check_schedule_feasibility)
graph_builder.add_node("send_to_reasoner", send_to_reasoner)
graph_builder.add_node("send_to_agent3", send_to_agent3)
graph_builder.add_node("rework_schedule", rework_schedule)

# Add edges
graph_builder.add_edge(START, "geocode_locations")
graph_builder.add_edge("geocode_locations", "find_nearby_locations")
graph_builder.add_edge("find_nearby_locations", "calculate_travel_times")
graph_builder.add_edge("calculate_travel_times", "create_schedule")
graph_builder.add_edge("create_schedule", "check_schedule_feasibility")
graph_builder.add_edge("check_schedule_feasibility", "send_to_reasoner")

# Add conditional edges
graph_builder.add_conditional_edges(
    "send_to_reasoner",
    lambda state: state.get("next"),
    {
        "send_to_agent3": "send_to_agent3",
        "rework_schedule": "rework_schedule"
    }
)

graph_builder.add_edge("send_to_agent3", END)
graph_builder.add_edge("rework_schedule", END)

# Compile the graph
graph = graph_builder.compile()

# Example data for testing
example_tasks = [
    TaskWithTime(
        task_id="task1",
        name="Buy groceries",
        duration_minutes=30,
        location=TaskLocation(
            task_id="task1",
            name="Grocery Store",
            is_fixed=False,
            address="Supermarket near downtown"
        )
    ),
    TaskWithTime(
        task_id="task2",
        name="Medical Center",
        duration_minutes=60,
        location=TaskLocation(
            task_id="task2",
            name="Medical Center",
            is_fixed=True,
            address="123 Medical Plaza, Mission District, San Francisco",
            lat=37.7599,  # Example latitude for Mission District, SF
            lng=-122.4279   # Example longitude for Mission District, SF
        )
    ),
    TaskWithTime(
        task_id="task3",
        name="School",
        duration_minutes=15,
        location=TaskLocation(
            task_id="task3",
            name="School",
            is_fixed=True,
            address="456 Education Lane, San Francisco",
            lat=37.7800,  # Reverted latitude for San Francisco
            lng=-122.4800   # Reverted longitude for San Francisco
        )
    )
]

# Example function to run the agent
def run_geolocator_agent(tasks=None):
    if tasks is None:
        tasks = example_tasks
    
    initial_state = {
        "messages": [],
        "tasks": tasks,
        "locations": [task.location for task in tasks],
        "travel_info": [],
        "schedule_feasible": None,
        "schedule_feedback": None
    }
    
    result = graph.invoke(initial_state)
    
    print("\nFinal Schedule:")
    for task in result.get("tasks", []):
        start_time = task.start_time.strftime("%H:%M") if task.start_time else "N/A"
        end_time = task.end_time.strftime("%H:%M") if task.end_time else "N/A"
        location_str = f"{task.location.name}"
        if task.location.lat and task.location.lng:
            location_str += f" ({task.location.lat:.4f}, {task.location.lng:.4f})"
        
        print(f"{start_time} - {end_time}: {task.name} at {location_str}")
    
    print("\nTravel Information:")
    for travel in result.get("travel_info", []):
        print(f"From {travel.from_task_id} to {travel.to_task_id}: {travel.travel_time_minutes} minutes ({travel.distance_km:.2f} km)")
    
    print(f"\nFeasibility: {result.get('schedule_feasible')}")
    print(f"Feedback: {result.get('schedule_feedback')}")
    
    return result

if __name__ == "__main__":
    run_geolocator_agent()
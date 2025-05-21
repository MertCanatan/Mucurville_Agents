# agent2.py — Geolocator Agent
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import logging
import os
import datetime
import time
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize geolocation services
geolocator = Nominatim(user_agent="schedule_optimizer")

# OSRM (Open Source Routing Machine) service URL
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

# Rate limiting - Nominatim has a strict usage policy
MIN_TIME_BETWEEN_REQUESTS = 1  # seconds
last_request_time = 0

# Define data models
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

class GeolocationResult(BaseModel):
    """Result of geolocation operations"""
    locations: List[Dict[str, Any]]
    travel_matrix: List[Dict[str, Any]]
    geocoding_errors: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

def geocode_address(address: str, name: str = "") -> Tuple[Optional[float], Optional[float], List[str]]:
    """
    Geocode an address using Nominatim geocoding service.
    
    Args:
        address: The address to geocode
        name: Optional name for logging
        
    Returns:
        Tuple of (lat, lng, errors)
    """
    global last_request_time
    
    if not address:
        return None, None, [f"No address provided for {name}" if name else "No address provided"]
    
    # Respect rate limiting
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < MIN_TIME_BETWEEN_REQUESTS:
        time.sleep(MIN_TIME_BETWEEN_REQUESTS - time_since_last)
    
    try:
        # Update last request time
        last_request_time = time.time()
        
        # Geocode the address
        location = geolocator.geocode(address, timeout=10, exactly_one=True)
        
        if location:
            logger.info(f"Geocoded address: {address} -> {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude, []
        else:
            logger.warning(f"Could not geocode address: {address}")
            return None, None, [f"Could not geocode address: {address}"]
    except Exception as e:
        error_msg = f"Error geocoding address {address}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, [error_msg]

def get_route_info(origin: Tuple[float, float], destination: Tuple[float, float]) -> Tuple[Optional[float], Optional[float], List[str]]:
    """
    Get route information between two points using OSRM.
    
    Args:
        origin: Tuple of (lat, lng) for the starting point
        destination: Tuple of (lat, lng) for the destination point
        
    Returns:
        Tuple of (distance_km, duration_min, errors)
    """
    try:
        # Format coordinates as required by OSRM: lon,lat
        origin_str = f"{origin[1]},{origin[0]}"
        dest_str = f"{destination[1]},{destination[0]}"
        
        # Make the request to OSRM
        url = f"{OSRM_BASE_URL}/{origin_str};{dest_str}?overview=false"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') != 'Ok':
            error_msg = f"OSRM API error: {data.get('message', 'Unknown error')}"
            logger.error(error_msg)
            return None, None, [error_msg]
        
        # Get the first route (should be the fastest)
        route = data['routes'][0]
        distance_km = route['distance'] / 1000  # Convert meters to km
        duration_min = route['duration'] / 60    # Convert seconds to minutes
        
        return distance_km, duration_min, []
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling OSRM API: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, [error_msg]
    except (KeyError, IndexError, ValueError) as e:
        error_msg = f"Error parsing OSRM response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, [error_msg]

def calculate_travel_matrix(locations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Calculate travel information between all pairs of locations using OSRM.
    
    Args:
        locations: List of location dictionaries with 'id', 'lat' and 'lng' keys
        
    Returns:
        Tuple of (travel_matrix, errors)
    """
    travel_matrix = []
    errors = []
    
    # First, ensure all locations have coordinates
    valid_locations = []
    for loc in locations:
        if 'lat' not in loc or 'lng' not in loc or loc['lat'] is None or loc['lng'] is None:
            errors.append(f"Location {loc.get('id', 'unknown')} is missing coordinates")
            continue
        valid_locations.append(loc)
    
    if len(valid_locations) < 2:
        errors.append("Need at least 2 valid locations to calculate travel matrix")
        return [], errors
    
    try:
        # Calculate travel info between all pairs of locations
        for i, loc1 in enumerate(valid_locations):
            for j, loc2 in enumerate(valid_locations):
                if i == j:
                    continue  # Skip self-comparison
                
                try:
                    from_id = loc1['id']
                    to_id = loc2['id']
                    
                    # Get coordinates
                    origin = (loc1['lat'], loc1['lng'])
                    destination = (loc2['lat'], loc2['lng'])
                    
                    # Calculate direct distance as fallback
                    direct_distance_km = geodesic(origin, destination).kilometers
                    
                    # Get route info from OSRM
                    distance_km, duration_min, route_errors = get_route_info(origin, destination)
                    
                    # Fall back to direct distance if OSRM fails
                    if distance_km is None or duration_min is None:
                        errors.extend(route_errors)
                        # Estimate driving time based on direct distance (assuming 30 km/h average speed)
                        distance_km = direct_distance_km
                        duration_min = (distance_km / 30) * 60  # Convert hours to minutes
                        logger.warning(f"Using estimated travel time for {from_id} -> {to_id}")
                    
                    # Get the task IDs from the location data if available
                    from_task_id = loc1.get('task_id', f'task_{from_id[-1]}')
                    to_task_id = loc2.get('task_id', f'task_{to_id[-1]}')
                    
                    travel_matrix.append({
                        'from_task_id': from_task_id,
                        'to_task_id': to_task_id,
                        'from_location_id': from_id,
                        'to_location_id': to_id,
                        'distance_km': distance_km,
                        'travel_time_min': round(duration_min, 1),  # Round to 1 decimal place
                        'travel_time_minutes': round(duration_min, 1),  # Keep both for compatibility
                        'mode': 'driving'  # Default mode
                    })
                    
                except Exception as e:
                    error_msg = f"Error calculating route between {loc1.get('id', 'unknown')} and {loc2.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
        
        logger.info(f"Calculated travel matrix for {len(valid_locations)} locations with {len(errors)} errors")
        
    except Exception as e:
        error_msg = f"Error in travel matrix calculation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    return travel_matrix, errors

def process_locations(locations: List[Dict[str, Any]]) -> GeolocationResult:
    """
    Process a list of locations by geocoding them and calculating travel times.
    
    Args:
        locations: List of location dictionaries with at least 'id' and 'address' keys
        
    Returns:
        GeolocationResult with processed data
    """
    if not locations:
        return GeolocationResult(locations=[], travel_matrix=[])
    
    # Make a copy to avoid modifying the input
    processed_locations = [loc.copy() for loc in locations]
    all_errors = []
    
    # Step 1: Geocode all locations
    for loc in processed_locations:
        if not loc.get('lat') or not loc.get('lng'):
            lat, lng, errors = geocode_address(loc.get('address'), loc.get('name', loc.get('id', 'Unknown')))
            if lat is not None and lng is not None:
                loc['lat'] = lat
                loc['lng'] = lng
            all_errors.extend(errors)
    
    # Step 2: Calculate travel matrix
    travel_matrix, matrix_errors = calculate_travel_matrix(processed_locations)
    all_errors.extend(matrix_errors)
    
    return GeolocationResult(
        locations=processed_locations,
        travel_matrix=travel_matrix,
        geocoding_errors=all_errors
    )

# Cache for storing nearby locations to avoid repeated API calls
NEARBY_LOCATIONS_CACHE = {}

def find_nearby_locations(location: Dict[str, Any], location_type: str = "cafe", radius_km: int = 1,
                          max_results: int = 3) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Find nearby locations for a given location using a simple distance-based approach.
    
    Note: This is a simplified implementation that searches for cafes by default.
    It uses a small cache to avoid hitting API limits.
    
    Args:
        location: Dictionary with 'lat' and 'lng' keys
        location_type: Type of location to search for (e.g., 'restaurant', 'cafe'). Defaults to 'cafe'.
        radius_km: Search radius in kilometers. Defaults to 1km to be more precise.
        max_results: Maximum number of results to return. Defaults to 3 to be conservative.
        
    Returns:
        Tuple of (list of nearby locations, list of errors)
    """
    global last_request_time, NEARBY_LOCATIONS_CACHE
    results = []
    errors = []
    
    # Create a cache key based on the location and type
    cache_key = f"{location['lat']:.4f},{location['lng']:.4f}_{location_type}"
    
    # Check cache first
    if cache_key in NEARBY_LOCATIONS_CACHE:
        logger.info(f"Using cached results for {location_type} near {location['lat']}, {location['lng']}")
        return NEARBY_LOCATIONS_CACHE[cache_key], []
    
    try:
        # Respect rate limiting
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_TIME_BETWEEN_REQUESTS:
            time.sleep(MIN_TIME_BETWEEN_REQUESTS - time_since_last)
        
        # Update last request time
        last_request_time = time.time()
        
        # Create a geolocator instance with longer timeout
        geolocator = Nominatim(
            user_agent="schedule_optimizer_app",  # More specific user agent
            timeout=5  # Reasonable timeout
        )
        
        # Get the address components for the location
        location_str = f"{location['lat']}, {location['lng']}"
        logger.info(f"Searching for {location_type} near {location_str}")
        
        # Use a more specific query for better results
        query = f"{location_type} near {location_str}" if location_type else f"point of interest near {location_str}"
        
        try:
            # Try to get nearby locations using Nominatim
            locations = geolocator.geocode(
                query,
                exactly_one=False,
                limit=max_results * 2,  # Get more than needed to filter by distance
                addressdetails=True,
                namedetails=True
            ) or []
            
            # If no results, try a more general search
            if not locations and location_type:
                logger.info(f"No results for '{query}', trying a more general search")
                locations = geolocator.geocode(
                    f"{location_type} near me",
                    exactly_one=False,
                    limit=max_results,
                    addressdetails=True,
                    namedetails=True
                ) or []
            
            # Filter and format results
            origin = (location['lat'], location['lng'])
            
            for loc in locations:
                try:
                    loc_point = (loc.latitude, loc.longitude)
                    dist = geodesic(origin, loc_point).kilometers
                    
                    if dist <= radius_km:
                        # Get the most relevant name from the response
                        name = (
                            getattr(loc, 'name', None) or 
                            (loc.raw.get('display_name', '').split(',')[0] if hasattr(loc, 'raw') else '') or 
                            f"Unnamed {location_type}"
                        )
                        
                        address = (
                            loc.raw.get('display_name', 'No address') 
                            if hasattr(loc, 'raw') else 'No address'
                        )
                        
                        results.append({
                            'name': name,
                            'address': address,
                            'lat': loc.latitude,
                            'lng': loc.longitude,
                            'distance_km': round(dist, 2)
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing location: {str(e)}")
                    continue
            
            # Sort by distance
            results = sorted(results, key=lambda x: x['distance_km'])
            
            # Cache the results
            NEARBY_LOCATIONS_CACHE[cache_key] = results
            
            logger.info(f"Found {len(results)} {location_type} locations near {location_str}")
            
        except Exception as e:
            error_msg = f"Error in location search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in find_nearby_locations: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    return results, errors

def get_directions(
    origin: Dict[str, float], 
    destination: Dict[str, float],
    mode: str = "driving"
) -> Dict[str, Any]:
    """
    Get directions between two points using OSRM.
    
    Args:
        origin: Dictionary with 'lat' and 'lng' keys
        destination: Dictionary with 'lat' and 'lng' keys
        mode: Travel mode (driving, walking, cycling)
        
    Returns:
        Dictionary with route information
    """
    # Validate input
    if 'lat' not in origin or 'lng' not in origin:
        raise ValueError("Origin must contain 'lat' and 'lng' keys")
    if 'lat' not in destination or 'lng' not in destination:
        raise ValueError("Destination must contain 'lat' and 'lng' keys")
    
    # Map mode to OSRM profile
    profile = 'foot' if mode == 'walking' else 'bike' if mode == 'cycling' else 'car'
    
    # Format coordinates as required by OSRM: lon,lat
    origin_str = f"{origin['lng']},{origin['lat']}"
    dest_str = f"{destination['lng']},{destination['lat']}"
    
    try:
        # Make the request to OSRM
        url = f"http://router.project-osrm.org/route/v1/{profile}/{origin_str};{dest_str}?overview=simplified&steps=true"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') != 'Ok':
            error_msg = f"OSRM API error: {data.get('message', 'Unknown error')}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Get the first route (should be the fastest)
        route = data['routes'][0]
        
        # Parse the steps
        steps = []
        for leg in route.get('legs', []):
            for step in leg.get('steps', []):
                steps.append({
                    'distance': {
                        'text': f"{step['distance'] / 1000:.1f} km",
                        'value': step['distance']
                    },
                    'duration': {
                        'text': f"{step['duration'] / 60:.0f} mins",
                        'value': step['duration']
                    },
                    'travel_mode': mode.upper(),
                    'start_location': {
                        'lat': step['maneuver']['location'][1],
                        'lng': step['maneuver']['location'][0]
                    },
                    'end_location': {
                        'lat': step['geometry']['coordinates'][-1][1],
                        'lng': step['geometry']['coordinates'][-1][0]
                    },
                    'html_instructions': step.get('name', 'Continue'),
                    'maneuver': step.get('maneuver', {}).get('modifier', '')
                })
        
        # Prepare the response
        return {
            'origin': origin,
            'destination': destination,
            'distance': {
                'text': f"{route['distance'] / 1000:.1f} km",
                'value': route['distance']
            },
            'duration': {
                'text': f"{route['duration'] / 60:.0f} mins",
                'value': route['duration']
            },
            'steps': steps,
            'overview_polyline': route.get('geometry'),
            'warnings': data.get('waypoints', []),
            'waypoint_order': [],
            'via_waypoint': []
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error getting directions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg) from e
    except (KeyError, IndexError, ValueError) as e:
        error_msg = f"Error parsing directions response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg) from e

def main():
    """Example usage of the geolocation functions"""
    # Example locations
    locations = [
        {
            'id': 'home',
            'name': 'Home',
            'address': '350 5th Ave, New York, NY 10118',
            'is_fixed': True
        },
        {
            'id': 'work',
            'name': 'Work',
            'address': '767 5th Ave, New York, NY 10153',
            'is_fixed': True
        },
        {
            'id': 'lunch',
            'name': 'Lunch Spot',
            'is_fixed': False,
            'address': '11 Madison Ave, New York, NY 10010'
        }
    ]
    
    # Process locations
    print("Processing locations...")
    result = process_locations(locations)
    
    # Print results
    print("\nGeocoding Results:")
    for loc in result.locations:
        print(f"- {loc['name']}: {loc.get('lat', 'N/A')}, {loc.get('lng', 'N/A')}")
    
    print("\nTravel Matrix:")
    for trip in result.travel_matrix:
        print(f"From {trip['from_location_id']} to {trip['to_location_id']}: "
              f"{trip['travel_time_min']} min, {trip['distance_km']} km")
    
    if result.geocoding_errors:
        print("\nErrors:")
        for error in result.geocoding_errors:
            print(f"- {error}")
    
    # Example of finding nearby locations
    if result.locations:
        print("\nFinding nearby coffee shops...")
        nearby, errors = find_nearby_locations(
            result.locations[0], 
            location_type='cafe',
            radius_km=1,
            max_results=3
        )
        
        print("\nNearby Coffee Shops:")
        for place in nearby:
            print(f"- {place['name']} ({place['distance_km']} km): {place['address']}")

if __name__ == "__main__":
    main()

def validate_locations(locations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that all required locations have valid coordinates.
    
    Args:
        locations: List of location dictionaries
        
    Returns:
        Dictionary with validation results
    """
    missing_coords = []
    invalid_coords = []
    
    for loc in locations:
        if 'lat' not in loc or 'lng' not in loc:
            missing_coords.append(loc.get('id', 'unknown'))
        elif not isinstance(loc['lat'], (int, float)) or not isinstance(loc['lng'], (int, float)):
            invalid_coords.append(loc.get('id', 'unknown'))
        elif not (-90 <= loc['lat'] <= 90) or not (-180 <= loc['lng'] <= 180):
            invalid_coords.append(loc.get('id', 'unknown'))
    
    is_valid = not (missing_coords or invalid_coords)
    
    return {
        'is_valid': is_valid,
        'missing_coordinates': missing_coords,
        'invalid_coordinates': invalid_coords,
        'message': (
            'All locations have valid coordinates.' if is_valid else
            f"Missing coordinates for: {', '.join(missing_coords)}. "
            f"Invalid coordinates for: {', '.join(invalid_coords)}"
        )
    }

# Export the main functionality
__all__ = [
    'geocode_address',
    'calculate_travel_matrix',
    'process_locations',
    'find_nearby_locations',
    'get_directions',
    'validate_locations',
    'GeolocationResult'
]

# Main execution for testing
if __name__ == "__main__":
    main()
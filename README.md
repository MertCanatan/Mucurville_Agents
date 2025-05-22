# Schedule Optimizer Multi-Agent System

A sophisticated multi-agent system for optimizing schedules using AI. This system consists of three specialized agents working together to create optimal schedules based on tasks, locations, and user preferences.

## Overview

The system is composed of three main agents:

1. **Planner Agent (Agent 1)**: Responsible for task planning and prioritization.
2. **Geolocator Agent (Agent 2)**: Handles location data, geocoding, and travel time calculations.
3. **Reasoner Agent (Agent 3)**: Optimizes the schedule based on constraints and preferences.

## Features

- **Task Management**: Define tasks with priorities, durations, and locations.
- **Intelligent Scheduling**: Automatically schedules tasks considering travel times and constraints.
- **Location Intelligence**: Geocoding of addresses and travel time calculations.
- **Multi-Agent Collaboration**: Agents work together in an iterative process to refine the schedule.
- **Flexible Configuration**: Customize constraints and preferences to suit different needs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Mucurville_Agents.git
   cd Mucurville_Agents
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   GMI_API_KEY=your_gmi_api_key_here
   ```

## Usage

### Running the Optimizer

```python
from main import run_schedule_optimizer

# Define your tasks and locations
tasks = [
    {
        "id": "task1",
        "name": "Morning Work",
        "description": "Important client meeting",
        "duration_minutes": 120,
        "location_id": "work",
        "priority": "high",
        "category": "work"
    },
    # Add more tasks...
]

locations = [
    {
        "id": "home",
        "name": "Home",
        "address": "350 5th Ave, New York, NY 10118",
        "is_fixed": True
    },
    # Add more locations...
]

# Run the optimizer
result = run_schedule_optimizer(
    tasks=tasks,
    locations=locations,
    user_preferences={
        "work_hours": {"start": "09:00", "end": "17:00"},
        "lunch_time": "12:00-13:00"
    },
    constraints={
        "max_work_hours_per_day": 8,
        "mandatory_break_after_hours": 4
    },
    max_iterations=3
)
```

### Command Line Interface

You can also run the example directly from the command line:

```bash
python main.py
```

## Agent Details

### Planner Agent (Agent 1)
- **Responsibilities**:
  - Task categorization
  - Priority assignment
  - Initial scheduling constraints
- **Input**: Raw task list, user preferences
- **Output**: Prioritized and categorized tasks

### Geolocator Agent (Agent 2)
- **Responsibilities**:
  - Address geocoding
  - Travel time calculations
  - Location validation
- **Input**: Task locations, addresses
- **Output**: Geocoded locations, travel matrix

### Reasoner Agent (Agent 3)
- **Responsibilities**:
  - Schedule optimization
  - Constraint satisfaction
  - Iterative improvement
- **Input**: Tasks, locations, travel info, constraints
- **Output**: Optimized schedule, feedback

## Configuration

### Environment Variables

- `GMI_API_KEY`: API key for the GMI LLM service

### Configuration Files

- `requirements.txt`: Python dependencies
- `.env`: Environment variables

## Example

See the `__main__` section in `main.py` for a complete working example with sample data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Development

### Project Structure

```
Mucurville_Agents/
├── agent1.py          # Planner Agent
├── agent2.py          # Geolocator Agent
├── agent3.py          # Reasoner Agent
├── main.py            # Main application logic
├── pyproject.toml     # Project configuration
├── README.md          # This file
└── requirements.txt   # Project dependencies
```

### Running the Application

Run the main application:

```bash
python3 main.py
```

### Development

To set up the development environment:

1. Install the package in development mode:
   ```bash
   pip3 install -e .
   ```

2. Install development dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## License

This project is licensed under the MIT License.

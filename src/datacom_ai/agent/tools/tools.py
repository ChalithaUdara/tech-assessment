import os
from typing import Optional, Type, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Mock implementations
class FlightSearchInput(BaseModel):
    origin: str = Field(description="The origin city airport code")
    destination: str = Field(description="The destination city airport code")
    departure_date: str = Field(description="The departure date (YYYY-MM-DD)")

class FlightSearchTool(BaseTool):
    name: str = "flight_search"
    description: str = "Search for flights between two cities."
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, origin: str, destination: str, departure_date: str) -> str:
        # Mock response
        return f"Found flights from {origin} to {destination} on {departure_date}: Flight NZ123 ($200), Flight JQ456 ($150)"

class WeatherInput(BaseModel):
    location: str = Field(description="The city name to get weather for")
    date: Optional[str] = Field(description="The date for the forecast")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the weather forecast for a location."
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str, date: Optional[str] = None) -> str:
        # Mock response
        return f"Weather in {location} is sunny, 25°C"

class AttractionsInput(BaseModel):
    location: str = Field(description="The city to find attractions in")
    budget: Optional[str] = Field(description="Budget constraint for attractions")

class AttractionsTool(BaseTool):
    name: str = "get_attractions"
    description: str = "Find tourist attractions in a location."
    args_schema: Type[BaseModel] = AttractionsInput

    def _run(self, location: str, budget: Optional[str] = None) -> str:
        # Mock response
        return f"Attractions in {location}: Sky Tower ($40), Viaduct Harbour (Free), War Memorial Museum ($25)"

def get_tools() -> List[BaseTool]:
    """
    Get all available tools.
    
    This function maintains backward compatibility while delegating to ToolFactory.
    
    Returns:
        List of all available tools
    """
    from datacom_ai.agent.tools.tool_factory import ToolFactory
    return ToolFactory.get_all_tools()

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
    tools = []

    # Flight Search: Amadeus or Mock
    if os.environ.get("AMADEUS_CLIENT_ID") and os.environ.get("AMADEUS_CLIENT_SECRET"):
        try:
            from amadeus import Client
            from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
            from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
            from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
            
            # Fix for Pydantic v2 "class not fully defined" error
            try:
                # We must pass the Client CLASS, not an instance.
                # The tools will instantiate their own client or use the one passed in constructor if supported,
                # but for type resolution, it needs the class.
                AmadeusToolkit.model_rebuild(_types_namespace={"Client": Client})
                AmadeusClosestAirport.model_rebuild(_types_namespace={"Client": Client})
                AmadeusFlightSearch.model_rebuild(_types_namespace={"Client": Client})
            except AttributeError:
                pass # model_rebuild is pydantic v2 only
            
            # We need to inject the Azure LLM into the tools, otherwise they default to ChatOpenAI
            # and try to hit the OpenAI API directly, failing with our Azure keys.
            from langchain_openai import AzureChatOpenAI
            
            llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
            )
            
            # The toolkit doesn't accept an LLM in init, but the individual tools do.
            # We can get the tools and then modify them or re-instantiate them.
            # Actually, AmadeusToolkit is just a wrapper. Let's instantiate tools manually if needed,
            # or see if we can pass it.
            # Looking at source, AmadeusToolkit doesn't take LLM.
            # But AmadeusClosestAirport DOES.
            
            toolkit = AmadeusToolkit()
            amadeus_tools = toolkit.get_tools()
            
            # Re-instantiate tools that need LLM with our Azure LLM
            final_tools = []
            for tool in amadeus_tools:
                if isinstance(tool, AmadeusClosestAirport):
                    # Re-create with LLM
                    final_tools.append(AmadeusClosestAirport(llm=llm))
                else:
                    final_tools.append(tool)
            
            tools.extend(final_tools)
        except ImportError:
            print("Warning: 'amadeus' package not found. Using mock flight search.")
            tools.append(FlightSearchTool())
        except Exception as e:
            print(f"Warning: Failed to initialize Amadeus toolkit: {e}. Using mock flight search.")
            tools.append(FlightSearchTool())
    else:
        tools.append(FlightSearchTool())

    # Weather: Mock (as requested, sticking to mock for now unless keys provided later)
    tools.append(WeatherTool())

    # Attractions: Google Places or Mock
    if os.environ.get("GPLACES_API_KEY"):
        try:
            # Use the new package to avoid deprecation warning
            from langchain_google_community import GooglePlacesTool
            tools.append(GooglePlacesTool())
        except ImportError:
            print("Warning: 'langchain-google-community' package not found. Using mock attractions tool.")
            tools.append(AttractionsTool())
        except Exception as e:
            print(f"Warning: Failed to initialize Google Places tool: {e}. Using mock attractions tool.")
            tools.append(AttractionsTool())
    else:
        tools.append(AttractionsTool())

    return tools

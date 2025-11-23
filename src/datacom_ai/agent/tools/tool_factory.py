"""Factory for creating agent tools."""

import os
from typing import List, Optional
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI

from datacom_ai.agent.tools.tools import (
    FlightSearchTool,
    WeatherTool,
    AttractionsTool
)
from datacom_ai.utils.logger import logger


class ToolFactory:
    """Factory for creating agent tools with consistent error handling."""

    @staticmethod
    def create_flight_tool(llm: Optional[AzureChatOpenAI] = None) -> List[BaseTool]:
        """
        Create flight search tool(s) (Amadeus or mock).

        Args:
            llm: Optional LLM client for Amadeus tools that require it

        Returns:
            List of flight search tool instances
        """
        if os.environ.get("AMADEUS_CLIENT_ID") and os.environ.get("AMADEUS_CLIENT_SECRET"):
            try:
                return ToolFactory._create_amadeus_tool(llm)
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to initialize Amadeus toolkit: {e}. Using mock flight search.")
                return [FlightSearchTool()]
        return [FlightSearchTool()]

    @staticmethod
    def _create_amadeus_tool(llm: Optional[AzureChatOpenAI] = None) -> List[BaseTool]:
        """
        Create Amadeus flight search tools.

        Args:
            llm: Optional LLM client for tools that require it

        Returns:
            List of Amadeus tool instances

        Raises:
            ImportError: If Amadeus packages are not available
        """
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
            # model_rebuild is pydantic v2 only, ignore if not available
            pass

        # Create LLM if not provided
        if llm is None:
            llm = ToolFactory._create_azure_llm()

        # Get tools from toolkit
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

        return final_tools if final_tools else [FlightSearchTool()]

    @staticmethod
    def _create_azure_llm() -> AzureChatOpenAI:
        """
        Create Azure OpenAI LLM client for tools.

        Returns:
            Configured AzureChatOpenAI instance
        """
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    @staticmethod
    def create_weather_tool() -> BaseTool:
        """
        Create weather tool (currently mock only).

        Returns:
            Weather tool instance
        """
        return WeatherTool()

    @staticmethod
    def create_attractions_tool() -> BaseTool:
        """
        Create attractions tool (Google Places or mock).

        Returns:
            Attractions tool instance
        """
        if os.environ.get("GPLACES_API_KEY"):
            try:
                from langchain_google_community import GooglePlacesTool
                return GooglePlacesTool()
            except ImportError:
                logger.warning("'langchain-google-community' package not found. Using mock attractions tool.")
                return AttractionsTool()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Places tool: {e}. Using mock attractions tool.")
                return AttractionsTool()
        return AttractionsTool()

    @staticmethod
    def get_all_tools(llm: Optional[AzureChatOpenAI] = None) -> List[BaseTool]:
        """
        Get all available tools.

        Args:
            llm: Optional LLM client for tools that require it

        Returns:
            List of all available tools
        """
        tools = []
        # create_flight_tool returns a list (may have multiple Amadeus tools)
        tools.extend(ToolFactory.create_flight_tool(llm))
        tools.append(ToolFactory.create_weather_tool())
        tools.append(ToolFactory.create_attractions_tool())
        return tools


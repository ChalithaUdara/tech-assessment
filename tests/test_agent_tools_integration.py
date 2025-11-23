import pytest
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from datacom_ai.agent.planning_agent import PlanningAgent

load_dotenv()

@pytest.mark.asyncio
async def test_agent_flight_search_tool_integration():
    """
    Verify that the agent correctly calls the flight_search tool when asked for flights.
    """
    agent = PlanningAgent()
    
    # Prompt specifically for flights to trigger the tool
    prompt = "Find me flights from Auckland (AKL) to Sydney (SYD) on 2023-12-25."
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    # Run the graph
    final_state = await agent.graph.ainvoke(initial_state)
    messages = final_state["messages"]
    
    # Check for tool calls
    tool_calls_found = False
    flight_tool_called = False
    
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_found = True
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "flight_search":
                    flight_tool_called = True
                    args = tool_call["args"]
                    # Verify arguments are roughly correct (flexible matching)
                    assert "AKL" in args.values() or "Auckland" in str(args.values())
                    assert "SYD" in args.values() or "Sydney" in str(args.values())
    
    assert tool_calls_found, "The agent should have made at least one tool call."
    assert flight_tool_called, "The agent should have called the 'flight_search' tool."

@pytest.mark.asyncio
async def test_agent_weather_tool_integration():
    """
    Verify that the agent correctly calls the get_weather tool when asked about weather.
    """
    agent = PlanningAgent()
    
    prompt = "What is the weather like in Wellington right now?"
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    final_state = await agent.graph.ainvoke(initial_state)
    messages = final_state["messages"]
    
    weather_tool_called = False
    
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "get_weather":
                    weather_tool_called = True
                    args = tool_call["args"]
                    assert "Wellington" in args.values()
    
    assert weather_tool_called, "The agent should have called the 'get_weather' tool."

@pytest.mark.asyncio
async def test_agent_attractions_tool_integration():
    """
    Verify that the agent correctly calls the get_attractions tool when asked for attractions.
    """
    agent = PlanningAgent()
    
    prompt = "What are some top tourist attractions in Queenstown?"
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    final_state = await agent.graph.ainvoke(initial_state)
    messages = final_state["messages"]
    
    attractions_tool_called = False
    
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "get_attractions":
                    attractions_tool_called = True
                    args = tool_call["args"]
                    assert "Queenstown" in args.values()
    
    assert attractions_tool_called, "The agent should have called the 'get_attractions' tool."

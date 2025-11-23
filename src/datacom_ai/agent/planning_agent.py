import json
from typing import Annotated, List, TypedDict, Union, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from datacom_ai.agent.tools.tools import get_tools
from datacom_ai.utils.logger import logger

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    itinerary: dict

# Define the Agent
class PlanningAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.tools = get_tools()
        # Initialize Azure OpenAI client
        # Note: In a real app, we'd inject this or load config properly
        import os
        self.base_llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
        self.llm = self.base_llm.bind_tools(self.tools)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("finalizer", self._finalizer_node)
        workflow.add_node("formatter", self._formatter_node)

        # Define edges
        workflow.set_entry_point("planner")
        
        workflow.add_conditional_edges(
            "planner",
            self._should_continue,
            {
                "continue": "tools",
                "finalize": "finalizer",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "planner")
        workflow.add_edge("finalizer", "formatter")
        workflow.add_edge("formatter", END)

        return workflow.compile()

    def _planner_node(self, state: AgentState):
        messages = state["messages"]
        logger.info(f"[STEP: PLANNER] processing {len(messages)} messages")
        
        # Add a system message to encourage reasoning if not present
        # We check if the first message is a system message, if not we prepend one
        
        system_message = SystemMessage(content="""You are an autonomous travel planning agent. 
You have access to tools for searching flights, weather, and attractions. 
You MUST use these tools to fetch real-time data. 
Do NOT hallucinate or make up flight details, weather, or specific attraction prices if you can use a tool to get them.
Before calling any tools, you must analyze the user's request and plan your actions. 
Think step-by-step and explain your reasoning. Your reasoning will be logged as a scratch-pad.""")
        
        # Check if we already have a system message at the start
        if messages and isinstance(messages[0], SystemMessage):
            # If the existing system message is different/generic, we might want to replace or append?
            # For now, let's assume the graph state just has conversation history.
            # We prepend our agent instructions.
            messages_with_system = [system_message] + messages[1:]
        else:
            messages_with_system = [system_message] + messages
        
        response = self.llm.invoke(messages_with_system)
        
        # Log the reasoning (content)
        if response.content:
            logger.info(f"AGENT REASONING: {response.content}")
            
        if response.tool_calls:
            logger.info(f"[STEP: PLANNER] Decided to call {len(response.tool_calls)} tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info("[STEP: PLANNER] No tool calls made")
            
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "finalize", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM made tool calls, continue to tools
        if last_message.tool_calls:
            logger.info("[DECISION] Continuing to TOOLS node")
            return "continue"
        
        # If the LLM didn't make tool calls, check if we have enough info to finalize
        # For simplicity, if no tool calls, we assume it's ready to finalize or it's just chatting
        # We need a way to detect if the plan is ready. 
        # Let's assume if it's not a tool call, it's the final response or a question.
        # But we want to force JSON output at the end.
        
        logger.info("[DECISION] Finalizing plan (no tool calls)")
        return "finalize"

    def _finalizer_node(self, state: AgentState):
        logger.info("[STEP: FINALIZER] Generating JSON itinerary")
        # This node ensures the output is formatted as JSON
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a travel assistant. Create a JSON itinerary based on the conversation history. The JSON should have keys: destination, duration, budget, activities, flight, accommodation, total_cost. If the conversation is not yet at a stage where a complete itinerary can be formed (e.g. asking for clarification, missing dates, missing destination), return an empty JSON object {{}}."),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | self.base_llm
        response = chain.invoke({"messages": messages})
        
        try:
            # Try to parse JSON to validate
            json_content = response.content
            # Clean up markdown code blocks if present
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()
                
            itinerary = json.loads(json_content)
            # We return the itinerary in the state, but we DO NOT add the JSON message to the conversation history
            # meant for the user. We will let the formatter do that.
            # However, we might want to keep it in history for debugging? 
            # Let's add it but we won't stream it as the final answer.
            logger.info("[STEP: FINALIZER] Successfully generated itinerary JSON")
            return {"itinerary": itinerary, "messages": [AIMessage(content=json.dumps(itinerary, indent=2))]}
        except:
            # Fallback if JSON parsing fails
            logger.warning("[STEP: FINALIZER] Failed to parse JSON, returning raw response")
            return {"messages": [response]}

    def _formatter_node(self, state: AgentState):
        logger.info("[STEP: FORMATTER] Formatting itinerary for user")
        # This node converts the JSON itinerary into a natural language response
        itinerary = state.get("itinerary")
        
        if not itinerary:
            logger.info("[STEP: FORMATTER] No itinerary found, skipping formatting")
            return {}
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful travel assistant. Present the following travel itinerary to the user in a friendly, engaging, and readable natural language format. Do not output JSON."),
            ("human", "{itinerary}")
        ])
        
        chain = prompt | self.base_llm
        response = chain.invoke({"itinerary": json.dumps(itinerary)})
        
        return {"messages": [response]}

    def stream(self, messages: List[BaseMessage]):
        # Wrapper to match ChatEngine interface
        # LangGraph stream yields events. We need to convert them to StreamUpdate.
        
        inputs = {"messages": messages}
        
        buffered_planner_content = None
        
        for event in self.graph.stream(inputs):
            for key, value in event.items():
                if key == "planner":
                    msg = value["messages"][0]
                    
                    # Log Thought (Content)
                    if msg.content:
                        logger.info(f"Thought: {msg.content}")
                        # Buffer the content. If we don't get a final itinerary, we will yield this.
                        buffered_planner_content = msg.content
                        
                    # Log Action (Tool Calls)
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            logger.info(f"Action: {tool_call['name']} ({tool_call['args']})")
                            
                    # If the message has tool calls, we can yield a status update or the content if any
                    if msg.tool_calls:
                        if msg.content:
                            yield {"content": msg.content}
                    else:
                        # If no tool calls, we wait for the finalizer/formatter
                        pass
                    
                    # Check for usage metadata
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        yield {"metadata": {"usage": msg.usage_metadata}}
                        
                elif key == "tools":
                    # Log Observation (Tool Outputs)
                    for msg in value["messages"]:
                        if isinstance(msg, ToolMessage):
                            logger.info(f"Observation: {msg.content}")
                            
                elif key == "finalizer":
                    # We do NOT yield the JSON content to the user stream usually
                    # But if it was a fallback (raw text), we might want to?
                    # Actually, let's just rely on formatter.
                    # But if formatter returns empty, we missed the finalizer output.
                    
                    if value and "messages" in value:
                        msg = value["messages"][0]
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            yield {"metadata": {"usage": msg.usage_metadata}}
                        
                        # If we are in fallback mode (no itinerary in state), we should yield this message
                        # But we don't have easy access to state here, just the delta.
                        # We can check if the content looks like JSON?
                        # Or we can modify finalizer to return a flag?
                        # For now, let's just be safe.
                        pass
                        
                elif key == "formatter":
                    # This is the final natural language response we want to show
                    if value and "messages" in value:
                        msg = value["messages"][0]
                        yield {"content": msg.content}
                        
                        # Check for usage metadata
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            yield {"metadata": {"usage": msg.usage_metadata}}
                    else:
                        # If formatter didn't produce messages (e.g. no itinerary), 
                        # we yield the buffered planner content (the question/clarification)
                        if buffered_planner_content:
                            yield {"content": buffered_planner_content}

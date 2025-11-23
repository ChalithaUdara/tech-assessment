import json
from typing import Annotated, List, TypedDict, Union, Literal, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from datacom_ai.agent.tools.tools import get_tools
from datacom_ai.utils.logger import logger
from datacom_ai.utils.structured_logging import log_agent_step, log_error, log_agent_execution
import time


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    itinerary: dict

# Define the Agent
class PlanningAgent:
    def __init__(
        self, 
        llm_client: Optional[AzureChatOpenAI] = None
    ):
        """
        Initialize the Planning Agent.
        
        Args:
            llm_client: Optional AzureChatOpenAI client. If not provided, will be created
                       using create_llm_client(). This allows for dependency injection
                       and better testability.
        """
        self.tools = get_tools()
        
        # Use dependency injection for LLM client
        if llm_client is None:
            from datacom_ai.clients.llm_client import create_llm_client
            self.base_llm = create_llm_client()
        else:
            self.base_llm = llm_client
        
        # Create a version with tools bound
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
        
        step_start = time.time()
        tool_calls_list = []
        status = "success"
        error_message = None
        
        try:
            # Add a system message to encourage reasoning if not present
            # We check if the first message is a system message, if not we prepend one
            
            # Get current date and time
            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H:%M:%S")
            current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            system_message = SystemMessage(content=f"""You are an autonomous travel planning agent. 
You have access to tools for searching flights, weather, and attractions. 
You MUST use these tools to fetch real-time data. 
Do NOT hallucinate or make up flight details, weather, or specific attraction prices if you can use a tool to get them.
Before calling any tools, you must analyze the user's request and plan your actions. 
Think step-by-step and explain your reasoning. Your reasoning will be logged as a scratch-pad.

IMPORTANT DATE RULES:
- Today's date and time is {current_datetime_str} (Date: {current_date}, Time: {current_time}). 
- You ONLY know today's date - do NOT assume or guess departure dates.
- If the user does NOT provide a departure date, you MUST ask them for it. Do NOT assume it's tomorrow or any other date.
- When the user mentions relative dates like "tomorrow", "next week", etc., calculate them from today's date ({current_date}).
- Do NOT make up or hallucinate dates - always ask the user if a departure date is missing.""")
            
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
                tool_calls_list = [{"name": tc.get("name"), "args": tc.get("args", {})} for tc in response.tool_calls]
                logger.info(f"[STEP: PLANNER] Decided to call {len(response.tool_calls)} tools: {[tc['name'] for tc in response.tool_calls]}")
            else:
                logger.info("[STEP: PLANNER] No tool calls made")
            
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            # Log structured agent step
            log_agent_step(
                step_name="planner",
                status=status,
                tool_calls=tool_calls_list if tool_calls_list else None,
                execution_time_ms=execution_time_ms,
                message_count=len(messages)
            )
            
            return {"messages": [response]}
        except Exception as e:
            status = "failure"
            error_message = str(e)
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            log_agent_step(
                step_name="planner",
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )
            
            log_error(
                error_type="agent_planner_error",
                error_message=error_message,
                context={"step": "planner", "message_count": len(messages)}
            )
            raise

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
        
        step_start = time.time()
        status = "success"
        error_message = None
        
        try:
            # Get current date and time for the finalizer
            current_datetime = datetime.now()
            current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            current_date = current_datetime.strftime("%Y-%m-%d")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a travel assistant. Create a JSON itinerary based on the conversation history. The JSON should have keys: destination, duration, budget, activities, flight, accommodation, total_cost. If the conversation is not yet at a stage where a complete itinerary can be formed (e.g. asking for clarification, missing departure date, missing destination), return an empty JSON object {{{{}}}}.\n\nIMPORTANT DATE RULES:\n- Today's date is {current_date} ({current_datetime_str}). Use this as the reference point for all date calculations.\n- Do NOT assume or guess departure dates. If the departure date is missing from the conversation, return an empty JSON object.\n- Only create an itinerary when all required information (destination, departure date, duration) is provided by the user."),
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
                
                execution_time_ms = int((time.time() - step_start) * 1000)
                
                log_agent_step(
                    step_name="finalizer",
                    status=status,
                    execution_time_ms=execution_time_ms,
                    has_itinerary=True
                )
                
                return {"itinerary": itinerary, "messages": [AIMessage(content=json.dumps(itinerary, indent=2))]}
            except Exception as json_error:
                # Fallback if JSON parsing fails
                status = "failure"
                error_message = f"JSON parsing failed: {str(json_error)}"
                logger.warning(f"[STEP: FINALIZER] Failed to parse JSON: {error_message}")
                
                execution_time_ms = int((time.time() - step_start) * 1000)
                
                log_agent_step(
                    step_name="finalizer",
                    status=status,
                    error_message=error_message,
                    execution_time_ms=execution_time_ms,
                    has_itinerary=False
                )
                
                return {"messages": [response]}
        except Exception as e:
            status = "failure"
            error_message = str(e)
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            log_agent_step(
                step_name="finalizer",
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )
            
            log_error(
                error_type="agent_finalizer_error",
                error_message=error_message,
                context={"step": "finalizer"}
            )
            raise

    def _formatter_node(self, state: AgentState):
        logger.info("[STEP: FORMATTER] Formatting itinerary for user")
        # This node converts the JSON itinerary into a natural language response
        itinerary = state.get("itinerary")
        
        step_start = time.time()
        status = "success"
        error_message = None
        
        if not itinerary:
            logger.info("[STEP: FORMATTER] No itinerary found, skipping formatting")
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            log_agent_step(
                step_name="formatter",
                status="success",
                execution_time_ms=execution_time_ms,
                has_itinerary=False
            )
            return {}
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful travel assistant. Present the following travel itinerary to the user in a friendly, engaging, and readable natural language format. Do not output JSON."),
                ("human", "{itinerary}")
            ])
            
            chain = prompt | self.base_llm
            response = chain.invoke({"itinerary": json.dumps(itinerary)})
            
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            log_agent_step(
                step_name="formatter",
                status=status,
                execution_time_ms=execution_time_ms,
                has_itinerary=True,
                response_length=len(response.content) if response.content else 0
            )
            
            return {"messages": [response]}
        except Exception as e:
            status = "failure"
            error_message = str(e)
            execution_time_ms = int((time.time() - step_start) * 1000)
            
            log_agent_step(
                step_name="formatter",
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )
            
            log_error(
                error_type="agent_formatter_error",
                error_message=error_message,
                context={"step": "formatter"}
            )
            raise

    def stream(self, messages: List[BaseMessage]):
        # Wrapper to match ChatEngine interface
        # LangGraph stream yields events. We need to convert them to StreamUpdate.
        
        # Track execution metrics for analytics
        execution_start = time.time()
        steps_executed = []
        steps_succeeded = 0
        steps_failed = 0
        failure_step = None
        failure_type = None
        tool_calls_count = 0
        tool_calls_succeeded = 0
        tool_calls_failed = 0
        has_itinerary = False
        overall_status = "success"
        
        inputs = {"messages": messages}
        
        buffered_planner_content = None
        
        try:
            for event in self.graph.stream(inputs):
                for key, value in event.items():
                    if key == "planner":
                        if "planner" not in steps_executed:
                            steps_executed.append("planner")
                        # Check if planner step succeeded (no exception means success)
                        # We'll track this from the log_agent_step calls, but for now assume success
                        # if we reach here
                        steps_succeeded += 1
                        
                        msg = value["messages"][0]
                        
                        # Log Thought (Content)
                        if msg.content:
                            logger.info(f"Thought: {msg.content}")
                            # Buffer the content. If we don't get a final itinerary, we will yield this.
                            buffered_planner_content = msg.content
                            
                        # Log Action (Tool Calls)
                        if msg.tool_calls:
                            tool_calls_count += len(msg.tool_calls)
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
                        if "tools" not in steps_executed:
                            steps_executed.append("tools")
                        steps_succeeded += 1
                        
                        # Log Observation (Tool Outputs)
                        tool_execution_start = time.time()
                        tool_results = []
                        
                        for msg in value["messages"]:
                            if isinstance(msg, ToolMessage):
                                logger.info(f"Observation: {msg.content}")
                                tool_results.append({
                                    "tool_call_id": msg.tool_call_id if hasattr(msg, "tool_call_id") else None,
                                    "content_length": len(msg.content) if msg.content else 0
                                })
                                tool_calls_succeeded += 1
                        
                        execution_time_ms = int((time.time() - tool_execution_start) * 1000)
                        
                        # Log tool execution step
                        log_agent_step(
                            step_name="tools",
                            status="success",
                            execution_time_ms=execution_time_ms,
                            tool_results_count=len(tool_results)
                        )
                                
                    elif key == "finalizer":
                        if "finalizer" not in steps_executed:
                            steps_executed.append("finalizer")
                        # Check if finalizer succeeded by checking if itinerary exists in state
                        # We'll check this from the log_agent_step has_itinerary field
                        # For now, assume success if we reach here
                        steps_succeeded += 1
                        
                        # We do NOT yield the JSON content to the user stream usually
                        # But if it was a fallback (raw text), we might want to?
                        # Actually, let's just rely on formatter.
                        # But if formatter returns empty, we missed the finalizer output.
                        
                        if value and "messages" in value:
                            msg = value["messages"][0]
                            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                yield {"metadata": {"usage": msg.usage_metadata}}
                            
                            # Check if itinerary was generated (check state if available)
                            if "itinerary" in value and value["itinerary"]:
                                has_itinerary = True
                            
                            # If we are in fallback mode (no itinerary in state), we should yield this message
                            # But we don't have easy access to state here, just the delta.
                            # We can check if the content looks like JSON?
                            # Or we can modify finalizer to return a flag?
                            # For now, let's just be safe.
                            pass
                            
                    elif key == "formatter":
                        if "formatter" not in steps_executed:
                            steps_executed.append("formatter")
                        steps_succeeded += 1
                        
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
            
            # Log agent execution summary after successful completion
            total_execution_time_ms = int((time.time() - execution_start) * 1000)
            log_agent_execution(
                agent_type="PlanningAgent",
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms,
                steps_executed=steps_executed,
                steps_succeeded=steps_succeeded,
                steps_failed=steps_failed,
                failure_step=failure_step,
                failure_type=failure_type,
                tool_calls_count=tool_calls_count,
                tool_calls_succeeded=tool_calls_succeeded,
                tool_calls_failed=tool_calls_failed,
                has_itinerary=has_itinerary
            )
            
        except Exception as e:
            # Log failure and re-raise
            overall_status = "failure"
            total_execution_time_ms = int((time.time() - execution_start) * 1000)
            
            # Try to determine which step failed based on last executed step
            if steps_executed:
                failure_step = steps_executed[-1]
                failure_type = f"agent_{failure_step}_error"
            else:
                failure_step = "unknown"
                failure_type = "agent_execution_error"
            
            steps_failed = 1
            
            log_agent_execution(
                agent_type="PlanningAgent",
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms,
                steps_executed=steps_executed,
                steps_succeeded=steps_succeeded,
                steps_failed=steps_failed,
                failure_step=failure_step,
                failure_type=failure_type,
                tool_calls_count=tool_calls_count,
                tool_calls_succeeded=tool_calls_succeeded,
                tool_calls_failed=tool_calls_failed,
                has_itinerary=has_itinerary,
                error_message=str(e)
            )
            raise

import subprocess
import os
import time
from typing import TypedDict, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from datacom_ai.utils.logger import logger
from datacom_ai.clients.llm_client import create_llm_client
from datacom_ai.utils.structured_logging import log_agent_execution

# 1. Defining the Agent State
class AgentState(TypedDict):
    task_prompt: str
    current_code: str
    filename: str
    execution_command: str
    test_output: str
    retry_count: int
    status_message: str
    thought: str

class SelfHealingAgent:
    def __init__(
        self, 
        llm_client: Optional[AzureChatOpenAI] = None
    ):
        """
        Initialize the Self-Healing Coding Agent.
        
        Args:
            llm_client: Optional AzureChatOpenAI client. If not provided, will be created
                       using create_llm_client(). This allows for dependency injection
                       and better testability.
        """
        if llm_client is None:
            self.llm = create_llm_client()
        else:
            self.llm = llm_client
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("generator", self._code_generator_node)
        workflow.add_node("executor", self._test_executor_node)

        # Define Edges
        workflow.set_entry_point("generator")
        workflow.add_edge("generator", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self._should_continue,
            {
                "success": END,
                "retry": "generator",
                "fail": END
            }
        )

        return workflow.compile()

    def _code_generator_node(self, state: AgentState):
        task = state["task_prompt"]
        test_output = state.get("test_output", "")
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"[Generator] Attempt {retry_count + 1}. Task: {task[:50]}...")

        if not test_output:
            # Initial generation
            system_prompt = """You are an expert coding assistant. Your task is to generate code to solve a specific problem.
You must output a JSON object with the following keys:
- "code": The full source code to solve the problem. Include any necessary imports and a test case if applicable (e.g. using `if __name__ == "__main__":`).
- "filename": The filename to save the code to (e.g. "solution.py").
- "execution_command": The command to run the code/tests (e.g. "python solution.py" or "pytest solution.py").
- "thought": Your reasoning process for generating this code. Explain your approach.

Ensure the code is complete and self-contained."""
            user_message = f"Task: {task}"
        else:
            # Refinement
            system_prompt = """You are an expert coding assistant. The previous code attempt failed.
Analyze the error output and generate a corrected version of the code.
You must output a JSON object with the following keys:
- "code": The full corrected source code.
- "filename": The filename to save the code to.
- "execution_command": The command to run the code/tests.
- "thought": Your reasoning process for fixing the code. Analyze the error and explain the fix.

Ensure you fix the specific errors reported."""
            user_message = f"Task: {task}\n\nPrevious Error Output:\n{test_output}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_message}")
        ])
        
        # Enforce JSON output
        chain = prompt | self.llm.with_structured_output(method="json_mode")
        
        try:
            response = chain.invoke({"user_message": user_message})
            # Handle case where response might be a dict or an object depending on with_structured_output implementation
            # Assuming it returns a dict because we asked for JSON
            if isinstance(response, dict):
                generated_code = response.get("code", "")
                filename = response.get("filename", "solution.py")
                execution_command = response.get("execution_command", f"python {filename}")
                thought = response.get("thought", "No thought provided.")
            else:
                # Fallback if it returns a pydantic object or something else (though json_mode usually returns dict)
                # For safety, let's assume it might be text if something goes wrong, but with_structured_output is robust.
                # If it's an object, we might need to access attributes.
                generated_code = getattr(response, "code", "")
                filename = getattr(response, "filename", "solution.py")
                execution_command = getattr(response, "execution_command", f"python {filename}")
                thought = getattr(response, "thought", "No thought provided.")

            status = "Generating initial code..." if not test_output else "Refining code based on error..."
            
            logger.info(f"[Generator] Thought: {thought}")
            logger.info(f"[Generator] Generated Code:\n{generated_code}")

            return {
                "current_code": generated_code,
                "filename": filename,
                "execution_command": execution_command,
                "status_message": status,
                "thought": thought
            }
        except Exception as e:
            logger.error(f"Error in generator: {e}")
            return {
                "status_message": f"Error generating code: {str(e)}",
                "current_code": "", # Stop if generation fails
                "retry_count": retry_count + 1 # Increment to avoid infinite loop if it keeps failing
            }

    def _test_executor_node(self, state: AgentState):
        code = state["current_code"]
        filename = state["filename"]
        command = state["execution_command"]
        retry_count = state.get("retry_count", 0)
        
        if not code:
            return {
                "test_output": "No code generated.",
                "retry_count": retry_count + 1,
                "status_message": "Generation failed."
            }

        logger.info(f"[Executor] Writing to {filename} and running: {command}")
        
        # Write to disk
        try:
            with open(filename, "w") as f:
                f.write(code)
        except Exception as e:
            return {
                "test_output": f"Failed to write file: {e}",
                "retry_count": retry_count + 1,
                "status_message": "File write error."
            }

        # Execute
        try:
            # Run the command with a timeout
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0
            
            status = "Tests passed!" if success else "Tests failed."
            
            logger.info(f"[Executor] Output:\n{output}")
            logger.info(f"[Executor] Status: {status}")

            return {
                "test_output": output if not success else "", # Clear error if success, or keep output? 
                # Logic says: "feeds them back to the model". If success, we don't need to feed back.
                # But we might want to keep the success output.
                # Let's store the output regardless, the conditional edge checks success.
                "test_output": output,
                "retry_count": retry_count + 1,
                "status_message": status
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_output": "Execution timed out.",
                "retry_count": retry_count + 1,
                "status_message": "Timeout."
            }
        except Exception as e:
            return {
                "test_output": f"Execution error: {e}",
                "retry_count": retry_count + 1,
                "status_message": "Execution error."
            }

    def _should_continue(self, state: AgentState) -> Literal["success", "retry", "fail"]:
        retry_count = state["retry_count"]
        test_output = state["test_output"]
        status = state["status_message"]
        
        # Check if successful
        # We can check return code if we stored it, or infer from status/output
        # In _test_executor_node we set status to "Tests passed!" if returncode == 0
        if status == "Tests passed!":
            return "success"
        
        if retry_count >= 3:
            return "fail"
        
        return "retry"

    def stream(self, task_prompt: str):
        # Track execution metrics for analytics
        execution_start = time.time()
        steps_executed = []
        steps_succeeded = 0
        steps_failed = 0
        failure_step = None
        failure_type = None
        overall_status = "success"
        
        inputs = {
            "task_prompt": task_prompt,
            "retry_count": 0,
            "test_output": "",
            "status_message": "Starting...",
            "thought": ""
        }
        
        yield f"[SYSTEM] Starting task: {task_prompt}"
        
        try:
            for event in self.graph.stream(inputs):
                for key, value in event.items():
                    node_name = key.upper()
                    
                    # Track steps executed
                    if key not in steps_executed:
                        steps_executed.append(key)
                    
                    # Check step status
                    status_message = value.get("status_message", "")
                    if "error" in status_message.lower() or "failed" in status_message.lower():
                        steps_failed += 1
                        failure_step = key
                        failure_type = f"agent_{key}_error"
                    elif "passed" in status_message.lower() or "success" in status_message.lower():
                        steps_succeeded += 1
                    else:
                        # Default to success if no explicit failure
                        steps_succeeded += 1
                    
                    # Log thought if present
                    if "thought" in value and value["thought"]:
                        yield f"[{node_name}] Thought: {value['thought']}"
                    
                    # Log status message
                    if "status_message" in value:
                        yield f"[{node_name}] {value['status_message']}"
                    
                    # Log output
                    if "test_output" in value and value["test_output"]:
                        # Yield output for visibility, maybe truncated
                        yield f"[{node_name}] Output:\n{value['test_output'][:500]}..."

                    # Log generated code
                    if "current_code" in value and value["current_code"]:
                        yield f"[{node_name}] Generated Code:\n```python\n{value['current_code']}\n```"
            
            # Determine overall status based on final state
            if steps_failed > 0:
                overall_status = "failure"
            elif len(steps_executed) > 0 and steps_succeeded == len(steps_executed):
                overall_status = "success"
            else:
                overall_status = "partial"
            
            # Log agent execution summary after completion
            total_execution_time_ms = int((time.time() - execution_start) * 1000)
            log_agent_execution(
                agent_type="SelfHealingAgent",
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms,
                steps_executed=steps_executed,
                steps_succeeded=steps_succeeded,
                steps_failed=steps_failed,
                failure_step=failure_step,
                failure_type=failure_type
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
                agent_type="SelfHealingAgent",
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms,
                steps_executed=steps_executed,
                steps_succeeded=steps_succeeded,
                steps_failed=steps_failed,
                failure_step=failure_step,
                failure_type=failure_type,
                error_message=str(e)
            )
            
            raise


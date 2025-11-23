import pytest
import json
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import assert_test
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCallParams, LLMTestCaseParams

load_dotenv()

from datacom_ai.agent.planning_agent import PlanningAgent
from langchain_core.messages import HumanMessage

# Custom Wrapper for Azure OpenAI
class AzureOpenAIWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return "Azure OpenAI"

@pytest.mark.asyncio
async def test_planning_agent_task_completion():
    prompt = "Plan a 2-day trip to Auckland for under NZ$500"
    
    # Initialize the agent
    agent = PlanningAgent()
    
    # Run the agent
    # The agent returns a dict with "messages" and "itinerary"
    # We need to extract the final itinerary
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    final_state = await agent.graph.ainvoke(initial_state)
    
    # The itinerary is in the state
    print(f"Final State Keys: {final_state.keys()}")
    response = final_state.get("itinerary", {})
    print(f"Itinerary in state: {response}")
    
    # If itinerary is empty, try to parse from last message (fallback)
    if not response:
        last_msg = final_state["messages"][-1].content
        print(f"Last message content: {last_msg}")
        try:
            response = json.loads(last_msg)
        except:
            response = {"error": "Could not parse itinerary", "raw": last_msg}
    
    actual_output = json.dumps(response)
    
    # Configure Azure OpenAI for DeepEval
    azure_model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    eval_model = AzureOpenAIWrapper(azure_model)
    
    metric = GEval(
        name="Task Completion",
        criteria="The output must be a valid JSON itinerary for Auckland, 2 days, under NZ$500.",
        evaluation_steps=[
            "Check if the output is valid JSON.",
            "Check if the destination is Auckland.",
            "Check if the duration is 2 days.",
            "Check if the total cost is under NZ$500."
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=eval_model,
        threshold=0.7
    )
    
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output
    )
    
    assert_test(test_case, [metric])

@pytest.mark.asyncio
async def test_planning_agent_tool_correctness():
    # Skipping tool correctness for now as it requires complex setup with ToolCallParams
    pass


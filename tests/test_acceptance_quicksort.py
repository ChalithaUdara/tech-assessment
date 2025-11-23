import pytest
import os
from datacom_ai.self_heal.agent import SelfHealingAgent

@pytest.mark.slow
def test_acceptance_quicksort():
    """
    Acceptance test for the Self-Healing Agent.
    Task: "Write a Python function to implement quicksort and a test case for it."
    """
    task = "Write a Python function to implement quicksort and a test case for it."
    agent = SelfHealingAgent()
    
    # Run the agent and collect logs
    logs = []
    for update in agent.stream(task):
        logs.append(update)
        print(update)
    
    # Check if the last log indicates success
    # The agent stream yields strings like "[EXECUTOR] Tests passed!"
    
    success = False
    for log in logs:
        if "Tests passed!" in log:
            success = True
            break
            
    assert success, "Agent failed to generate working quicksort code."
    
    # Clean up generated file if it exists
    # The agent usually names it based on the LLM output, often "quicksort.py" or "solution.py"
    # We can try to find it or just leave it for manual inspection if needed.
    # For a clean test, we should probably try to clean up.
    # But since we don't know the exact filename without parsing logs, we might skip strict cleanup or try common names.
    
    possible_files = ["quicksort.py", "solution.py", "quick_sort.py"]
    for f in possible_files:
        if os.path.exists(f):
            os.remove(f)

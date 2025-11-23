import pytest
from unittest.mock import MagicMock, patch
from datacom_ai.self_heal.agent import SelfHealingAgent, AgentState

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm

def test_retry_logic(mock_llm):
    """
    Test that the agent retries when tests fail.
    We will mock the LLM to return code, and mock subprocess.run to fail the first time and pass the second.
    """
    # Create a mock for the chain (result of with_structured_output)
    mock_chain = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    
    # Mock the __call__ method of the chain (since prompt | chain calls chain(input))
    # First attempt: returns code that fails
    # Second attempt: returns code that passes
    mock_chain.side_effect = [
        {
            "code": "print('fail')",
            "filename": "fail.py",
            "execution_command": "python fail.py",
            "thought": "First attempt"
        },
        {
            "code": "print('success')",
            "filename": "success.py",
            "execution_command": "python success.py",
            "thought": "Second attempt"
        }
    ]
    
    agent = SelfHealingAgent(llm_client=mock_llm)
    
    # Mock subprocess.run and open
    with patch("subprocess.run") as mock_run, \
         patch("builtins.open", new_callable=MagicMock) as mock_open:
        
        # First run: fails
        # Second run: passes
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="Error"),
            MagicMock(returncode=0, stdout="Success", stderr="")
        ]
        
        # Run the agent
        task = "Write code that eventually succeeds"
        logs = list(agent.stream(task))
        
        # Verify retries occurred
        # We expect:
        # 1. Generator (Attempt 1)
        # 2. Executor (Fail)
        # 3. Generator (Attempt 2)
        # 4. Executor (Success)
        
        # Check logs for retry evidence
        retry_logs = [log for log in logs if "Refining code based on error" in log]
        assert len(retry_logs) == 1, "Should have refined code once."
        
        success_logs = [log for log in logs if "Tests passed!" in log]
        assert len(success_logs) == 1, "Should have eventually passed."
        
        # Verify chain was called twice
        assert mock_chain.call_count == 2

import argparse
import sys
from dotenv import load_dotenv
from datacom_ai.self_heal.agent import SelfHealingAgent
from datacom_ai.utils.logger import logger

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Self-Healing Code Assistant")
    parser.add_argument("task", type=str, help="The natural language coding task (e.g., 'Write quicksort in Python')")
    
    args = parser.parse_args()
    
    print(f"Starting Self-Healing Agent with task: {args.task}")
    
    try:
        agent = SelfHealingAgent()
        
        print("-" * 50)
        for update in agent.stream(args.task):
            print(update)
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import json

from agent import Agent


class QueryService:

    def __init__(self):
        self.agent = Agent()

    def _create_query_session(self):
        self.agent = Agent()

    def ask_agent(self, question: str):
        try:
            answer = self.agent.query_agent(user_input=question)
            response_formatted = json.loads(answer, strict=False)
        except Exception as e:
            print(f"An error occurred when querying the agent. Error: {e}")
            raise e

        return {"response": response_formatted, }

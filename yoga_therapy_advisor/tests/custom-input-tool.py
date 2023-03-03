
# custom tool experiment

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

# Load the tool configs that are needed.
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)


class CustomInputTool(BaseTool):
    name = "Input"
    description = "useful for when you need ask the user for input. Do this step by step. Ask the user at every step."

    def _run(self, query: str) -> str:
        """Use the tool."""
        return input(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not supported yet")
    
class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return llm_math_chain.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not supported yet")



tools = [CustomInputTool(), CustomCalculatorTool()]

# construct the agent. we'll use the default agent type here
agent = initialize_agent(tools=tools, llm=llm, agent='zero-shot-react-description', verbose=True)


agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age?")

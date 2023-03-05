
# custom tool experiment

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Load the tool configs that are needed.
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

class CustomSearchTool(BaseTool):                                                                                                                               
    name = "Search"                                                                                                                                             
    description = "useful for when you need to answer questions about current events. Make this as exact match."                                                    
                                                                                                                                                                
    def _run(self, query: str) -> str:                                                                                                                          
        """Use the tool."""         
        response = search.run(query)
        print(f'The search response: {response}')                                                                                                                            
        return response                                                                                                                                
                                                                                                                                                                
    async def _arun(self, query: str) -> str:                                                                                                                   
        """Use the tool asynchronously."""                                                                                                                      
        raise NotImplementedError("BingSearchRun does not support async")              

class CustomInputTool(BaseTool):
    name = "Ask User Input"
    description = "useful for when you cound nit find the answer anywhere and you need ask the user for input."

    def _run(self, query: str) -> str:
        """Ask for input from user."""
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


tools = [CustomInputTool(), 
         CustomCalculatorTool(),
         # CustomSearchTool(),
        ]

# construct the agent. we'll use the default agent type here
agent = initialize_agent(tools=tools, llm=llm, agent='zero-shot-react-description', verbose=True)


# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
# agent.run("Who is Praveen Moudgal's wife? Print her name. What is her current age raised to the 0.43 power?")
agent.run("Answer questions step by step. Who is Praveen Moudgal's wife? What is her current age?")
# agent.run("Answer questions step-by-step. Who is Aurobindo Tripathy's wife? What is her current age?")
# agent.run("Answer questions step-by-step. Aurobindo Tripathy is not a celebrety. Who is Aurobindo Tripathy's wife? What is her current age?")
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age?")

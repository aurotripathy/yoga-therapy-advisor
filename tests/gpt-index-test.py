from gpt_index import GPTListIndex, SimpleWebPageReader, BeautifulSoupWebReader, GPTSimpleVectorIndex
# from IPython.display import Markdown, display
from langchain.agents import load_tools, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, LLMChain

Input_URL = "https://docs.custody.coinbase.com/#introduction" 
documents = SimpleWebPageReader(html_to_text=True).load_data([Input_URL])
index = GPTSimpleVectorIndex(documents)
index.save_to_disk("./doc_qa.json")
index2 = GPTSimpleVectorIndex.load_from_disk("./doc_qa.json")

# Creating your Langchain Agent
def querying_db(query: str):
  response = index2.query(query, verbose=True)
  return response

tools = [
    Tool(
        name = "QueryingDB",
        func=querying_db,
        description="This function takes a query string as input and returns the most relevant answer from the documentation as output"
    )]
llm = OpenAI(temperature=0)

# Test your agent by asking it a question
query_string = "What is coinbase custody?" #@param {type:"string"}

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
result = agent.run(query_string)
print(result)

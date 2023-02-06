from gpt_index import Document
from gpt_index import GPTListIndex, SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import load_tools, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, LLMChain


text_list_to_be_indexed = ['/media/auro/Dropbox/My Ideas/yoga advisor/OCRed/Chapters/Chapter 1 - Therapeutic Yoga and Its Essentials.docx',]

documents = [Document(t) for t in text_list_to_be_indexed]

# documents = SimpleDirectoryReader(file_to_be_indexed).load_data()

index = GPTSimpleVectorIndex(documents)

index_file = "../indexes/yoga_qa.json"
#save
index.save_to_disk(index_file)

#load
index2 = GPTSimpleVectorIndex.load_from_disk(index_file)

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
query_string = "defference between therapy youga and modern medicine?" #@param {type:"string"}

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
result = agent.run(query_string)
print(result)

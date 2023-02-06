from gpt_index import Document
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI


text_list_to_be_indexed = ['/media/auro/Dropbox/My Ideas/yoga advisor/OCRed/Chapters/Chapter 1 - Therapeutic Yoga and Its Essentials.docx',]

documents = [Document(t) for t in text_list_to_be_indexed]

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
# query_string = "What's the difference between yoga therapy and modern medicine?" 
query_string = "What are the various food categories in yoga?" 

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
result = agent.run(query_string)
print(result)

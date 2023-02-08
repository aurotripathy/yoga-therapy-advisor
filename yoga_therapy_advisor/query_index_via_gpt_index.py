import logging
import sys

from gpt_index import Document
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

# set Logging to DEBUG for more detailed outputs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from pudb import set_trace

index_file = "../indexes/yoga_qa.json"
index2 = GPTSimpleVectorIndex.load_from_disk(index_file)

# Creating your Langchain Agent
def querying_db(query: str):
  response = index2.query(query, verbose=True)
  return response

tools = [
    Tool(
        name = "GPT Index",
        func=lambda q: str(index.query(q)),
        description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
        return_direct=True
    ),
]

set_trace()
memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)


# Test your agent by asking it a question
query_string = "What's the difference between yoga therapy and modern medicine?" 
# query_string = "What are the various food categories in yoga?" 

result = agent_chain.run(query_string)
print(result)

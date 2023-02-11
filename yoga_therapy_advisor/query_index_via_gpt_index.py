import logging
import sys

# This goes on the top?
# set Logging to DEBUG for more detailed outputs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# from gpt_index import Document
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from pudb import set_trace


index_file = "../vector-indexes/yoga_qa.json"
index = GPTSimpleVectorIndex.load_from_disk(index_file)

tools = [
    Tool(
        name = "GPT Index",
        func=lambda q: str(index.query(q)),
        description="useful for when you want to answer questions from indexed text. Always, you must try the index first.",
        return_direct=True
    ),
]

# set_trace()
memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, verbose=True, agent="conversational-react-description", memory=memory)


# Test your agent by asking it a question
# query_string = "What's the difference between yoga therapy and modern medicine?" 
# query_string = "What are the various food categories in yoga?" 


# query_string = "Hello, my name is Bob. Remember it." 
# result = agent_chain.run(query_string)

query_string = "From the text provided, what are the three food categories in yoga?" 
# query_string = "From the text provided, How do I cure my stomach ache?" 
# set_trace()

print(f'What the query to index returns, executed independently:\n{index.query(query_string)}')
      
result = agent_chain.run(query_string)

# query_string = "What is Sattvic food?" 
# result = agent_chain.run(query_string)

# query_string = "What's me name?" 
# result = agent_chain.run(query_string)



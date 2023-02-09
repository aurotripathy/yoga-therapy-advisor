import logging
import sys

from gpt_index import Document
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from pudb import set_trace

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

folder_to_be_indexed = '/media/auro/Dropbox/My Ideas/yoga advisor/OCRed/for_indexing'
documents = SimpleDirectoryReader(folder_to_be_indexed).load_data()

set_trace()

index = GPTSimpleVectorIndex(documents)

index_file = "../vector-indexes/yoga_qa.json"
#save
index.save_to_disk(index_file)
print(f'Saved to {index_file}')



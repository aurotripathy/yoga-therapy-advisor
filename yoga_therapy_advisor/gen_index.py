from utils import (
    parse_docx,
    parse_pdf,
    parse_txt,
    search_docs,
    text_to_docs,
    get_answer,
    get_sources,
    wrap_text_in_html,
)
from langchain.vectorstores import FAISS, VectorStore
from langchain.docstore.document import Document
from typing import List, Dict, Any
from langchain.embeddings import OpenAIEmbeddings
import os
# from pudb import set_trace

openai_api_key = os.environ["OPENAI_API_KEY"]

def simple_embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index
       Deals with the OpenAI API key externally (ebv var)
    """

    # Embed the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # type: ignore
    index = FAISS.from_documents(docs, embeddings)

    return index


file_to_be_indexed = '/media/auro/Dropbox/My Ideas/yoga advisor/OCRed/Chapters/Chapter 1 - Therapeutic Yoga and Its Essentials.docx'

text = text_to_docs(file_to_be_indexed)
print(text)

index = simple_embed_docs(text)
print(index)

print(f'saving...')
index.save_local('./saved_index')



print(f'loading...')
index.load_local('./saved_index')




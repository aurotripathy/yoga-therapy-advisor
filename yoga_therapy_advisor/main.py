import os
import openai
import streamlit as st

from openai.error import OpenAIError
from langchain.vectorstores import FAISS

from gpt_index import (
    GPTListIndex,
    SimpleWebPageReader,
    BeautifulSoupWebReader,
    GPTSimpleVectorIndex
)

# from IPython.display import Markdown, display
from langchain.agents import load_tools, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, LLMChain

openai.api_key = os.environ["OPENAI_API_KEY"]

def clear_submit():
    st.session_state["submit"] = False


def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

# Creating your Langchain Agent
def querying_db(query: str):
  response = index.query(query, verbose=True)
  return response


st.set_page_config(page_title="YogaTherapyAdvisor", page_icon="🧘", layout="wide")
st.header("🧘Yoga Therapy Advisor")

with st.sidebar:
    st.markdown("# About")
    st.header(
        "🙋Ask questions about what ails you "
        "and get instant answers from "
        "trusted sources with citations"
    )
    st.markdown(
        "This tool is a work in progress. "
        "You can contribute to the project on [GitHub](https://github.com/aurotripathy/yoga_therapy_advisor) "
        "with your feedback and suggestions💡"
    )
    st.markdown("---")
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"
        "2. Ask a question what ails you💬\n"
    )

    api_key_input = True  # Forced
    set_openai_api_key(openai.api_key)
    
    st.markdown("---")
    st.markdown("Made by [mmz_001](https://twitter.com/mm_sasmitha)")


print(f'Loading the indexes...')
index = GPTSimpleVectorIndex.load_from_disk("../tests/doc_qa.json")
print(f'Done loading the indexes.')
st.session_state["api_key_configured"] = True


query = st.text_area("Ask a question about what ails you", on_change=clear_submit)
with st.expander("Advanced Options"):
    show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if show_full_doc and doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not st.session_state.get("api_key_configured"):
        st.error("Please configure your OpenAI API key!")
    elif not index:
        st.error("Please upload a document!")
    elif not query:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)

        tools = [
            Tool(
                name = "QueryingDB",
                func=querying_db,
                description="This function takes a query string \
                as input and returns the most relevant answer \
                from the documentation as output"
            )]
        llm = OpenAI(temperature=0)


        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
        result = agent.run(query)
        print(result)

        try:
            with answer_col:
                st.markdown("#### Answer")
                st.markdown(result)

        except OpenAIError as e:
            st.error(e._message)

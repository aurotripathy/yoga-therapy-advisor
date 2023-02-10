import os
import openai
import streamlit as st
from streamlit_chat import message

from openai.error import OpenAIError

from gpt_index import (
    GPTListIndex,
    GPTSimpleVectorIndex
)

from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


vector_index_file = "../vector-indexes/yoga_qa.json"

st.set_page_config(page_title="YogaTherapyAdvisor", page_icon="🧘", layout="wide")
st.header("🧘Yoga Therapy Advisor")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def set_openai_api_key():
    st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]


def gen_sidebar():
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
        set_openai_api_key()

        st.markdown("---")
        st.markdown("Made by [auro tripathy](https://www.linkedin.com/in/aurotripathy/)")

    
# Creating your Langchain Agent
def querying_db(query: str):
  # response = index.query(query, verbose=True)
  response = index.query(query)
  return response
    

def get_text():
    c1, c2, c3 = st.columns([1, 4, 4])  # This moves the input box to the right
    with c3:
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text 


#main
gen_sidebar()

print(f'Loading the indexes...')
index = GPTSimpleVectorIndex.load_from_disk(vector_index_file)
print(f'Done loading the indexes.')

st.session_state["api_key_configured"] = True

query = get_text()

if query:
    st.session_state["submit"] = True
    # Output Columns
    answer_col, sources_col = st.columns(2)

    tools = [
        # Tool(
        #     name = "QueryingDB",
        #     func=querying_db,
        #     description="This function takes a query string \
        #     as input and returns the most relevant answer \
        #     from the documentation as output"
        # )]
        Tool(
            name = "QueryingDB",
            func=querying_db,
            description="Always do this function first. This function takes a query string \
            as input and returns the most relevant answer \
            from the documentation as output"
        )]
    llm = OpenAI(temperature=0)


    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    result = agent.run(query)
    print(f'Result: {result}')

    
    st.session_state.past.append(query)
    st.session_state.generated.append(result)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


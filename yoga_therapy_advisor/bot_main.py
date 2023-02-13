import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

import os
import streamlit as st
from streamlit_chat import message

from openai.error import OpenAIError

from gpt_index import (
    GPTListIndex,
    GPTSimpleVectorIndex
)

from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
# from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory


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

    
def get_text():
    c1, c2, c3 = st.columns([1, 4, 4])  # This moves the input box to the right
    with c3:
        input_text = st.text_input("You: ", placeholder="Ask me a question about yoga therapy and health.", key="input")        
    return input_text 


#main
gen_sidebar()

logger.info(f'Loading the indexes...')
vector_index_file = "../vector-indexes/yoga_qa.json"
index = GPTSimpleVectorIndex.load_from_disk(vector_index_file)
logger.info(f'Done loading the indexes.')

st.session_state["api_key_configured"] = True

query = get_text()

if query:
    st.session_state["submit"] = True
    # Output Columns
    answer_col, sources_col = st.columns(2)

    tools = [
        Tool(
            name = "GPT Index",
            func=lambda q: str(index.query(q)),
            description="Useful for when you want to answer questions about yoga. As an assistant, you must always use this tool first, no exceptions.",
            return_direct=True
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)


    agent = initialize_agent(tools, llm, agent="conversational-react-description",
                             memory=memory,
                             verbose=True)
    result = agent.run(query)
    print(f'Result: {result}')

    
    st.session_state.past.append(query)
    st.session_state.generated.append(result)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


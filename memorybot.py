"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Avratanu Biswas
# Date: March 11, 2023
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain import LLMMathChain
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.agents import tool

API_O = st.secrets["OPENAI_API_KEY"]

llm_math = OpenAI(temperature=0,
            openai_api_key=API_O, 
            model_name='gpt-3.5-turbo', 
            verbose=False) 

llm_math_chain = LLMMathChain(llm=llm_math, verbose=False)
wikipedia = WikipediaAPIWrapper()

@tool
def get_abc_news_titles():
    """Returns the headlines of the latest news articles from ABC News Australia"""
    url = "https://www.abc.net.au/news/feed/2942460/rss.xml"
    response = requests.get(url)
    xml_data = response.text

    root = ET.fromstring(xml_data)
    titles_url = []
    

    for item in root.findall('.//item'):
        title = item.find('title').text
        url = item.find('link').text
        titles_url.append({'title': title, 'url': url})

    return titles_url

@tool
def get_abc_news_text(url):
    """Returns the text of an article from ABC News when given the articles url"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article = soup.find('article')
    text = article.get_text()
    return text

# Initialize Conversational Agent
tools = [
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="useful for when you need to answer questions that wikipedia may be able to answer"
    ),
    Tool(
        name="ABC News Headlines",
        func=get_abc_news_titles,
        description="useful for when you are asked about the current news. Returns the headlines of the latest news articles from ABC News"
    ),
    Tool(
        name="ABC News Article",
        func=get_abc_news_text.run,
        description="useful for loading a specific article from ABC News"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )

]

# Set Streamlit page configuration
st.set_page_config(page_title='Theodore', page_icon='🧐', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
#with st.sidebar.expander("🛠️ ", expanded=False):
#    # Option to preview memory store
#    if st.checkbox("Preview memory store"):
#        with st.expander("Memory-Store", expanded=False):
#            st.session_state.entity_memory.store
#    # Option to preview memory buffer
#    if st.checkbox("Preview memory buffer"):
#        with st.expander("Bufffer-Store", expanded=False):
#            st.session_state.entity_memory.buffer
#K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
st.title("Anne's Chatbot Companion Theodore 🧐")

# Ask the user to enter their OpenAI API key
API_O = st.secrets["OPENAI_API_KEY"]

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0.5,
                openai_api_key=API_O, 
                model_name='gpt-3.5-turbo', 
                verbose=True) 


    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=6 )
        
        # Create the ConversationChain object with the specified configuration
    agent_chain = initialize_agent(tools, llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=st.session_state.entity_memory, chat_history = st.session_state.entity_memory)
    #Conversation = ConversationChain(
    #        llm=llm, 
    #        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    #        memory=st.session_state.entity_memory
    #    )  
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = agent_chain.run(input=user_input, chat_history=st.session_state.entity_memory.buffer)
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="👩🏻‍🦱")
        st.success(st.session_state["generated"][i], icon="🧐")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
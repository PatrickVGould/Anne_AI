import streamlit as st
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper

# Initialize Wikipedia API Wrapper
wikipedia = WikipediaAPIWrapper()

# Initialize Language Model and Memory
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(temperature=0, api_key=openai_api_key)
memory = ConversationKGMemory(llm=llm, return_messages=True)

# Initialize Conversation Knowledge Graph Memory
memory.save_context({"input": "say hi to sam"}, {"output": "who is sam"})
memory.save_context({"input": "sam is a friend"}, {"output": "okay"})

# Initialize Prompt Template
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Initialize ConversationChain
conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=memory
)

# Initialize Conversational Agent
tools = [
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="useful for when you need to answer questions that wikipedia may be able to answer"
    ),
]
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# Define function to run conversational agent
def run_conversational_agent(input_text):
    # Update the memory with the new input
    memory.load_memory_variables({"input": input_text})
    # Get the response from the conversational agent
    response = agent_chain.run(input=input_text)
    return response

st.title("Fred the AI Chat Bot with Knowledge Graph Memory")
st.subheader("For Anne to talk to an AI with access to Wikipedia, ABC news, and other sources. In the personality of Mr. Fred Rogers")

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""

# Get the user input
user_input = st.text_input("You:", st.session_state["input"], key="input",
                           placeholder="Type your message here...", label_visibility='hidden')

# Generate the output using the run_conversational_agent function and the user input, and add the input/output to the session
if st.button("Send"):
    response = run_conversational_agent(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

# Display the conversation history using an expander
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(f"User: {st.session_state['past'][i]}", icon="👩🏻‍🦱")
        st.success(f"Fred: {st.session_state['generated'][i]}", icon="🧑")
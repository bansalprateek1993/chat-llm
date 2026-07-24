from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

st.title('Agentic chatbot with Langgraph')


CONFIG = {'configurable':{'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')
if user_input:
    # Add the message in message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)
    
    # response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]}, config=CONFIG)
    
    # ai_message = response['messages'][-1].content

    # For streaming assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = 'messages'
            )
        )
    
    # Storing the AI message to message history.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})


# st.write('User', user_input)

# thread_id = "1" # Unique id for users - different head id for the user
# config = {'configurable': {'thread_id': thread_id}}

# response = chatbot.invoke({'messages':[HumanMessage(content="What is python")]}, config=config)
# print(response['messages'][-1].content)

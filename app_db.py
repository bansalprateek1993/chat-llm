from agentic_chatbot_db_backend import chatbot, get_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid

# Generate the unique thread id
def generate_thread_id():
    return str(uuid.uuid4())

# Add new thread it to the conversaiton list
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Reseting a chat,once users click on the new chat button
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable":{
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])
    
st.title('Agentic chatbot with Langgraph')

# Create message history when app runs first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

add_thread(st.session_state["thread_id"])

st.sidebar.title("Previous Conversations")
# Add sidebar
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


# Display all conversatioon threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:
    # Create one sidebar for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_message = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            
            elif isinstance(message, AIMessage):
                role = "assistant"
            
            else:
                continue
        
            temp_message.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_message
        # Return the application to display the loaded message
        st.rerun()

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

    CONFIG = {'configurable':{'thread_id': st.session_state["thread_id"]},
              "metadata": {
                  "thread_id": st.session_state["thread_id"]
              },
              "run_name": "chat_trace"
    }

    # For streaming assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = 'messages'
            )
            if isinstance(message_chunk, AIMessage)
        )

    # Storing the AI message to message history.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # with st.chat_message('assistant'):
    #     st.text(ai_message)


# st.write('User', user_input)

# thread_id = "1" # Unique id for users - different head id for the user
# config = {'configurable': {'thread_id': thread_id}}

# response = chatbot.invoke({'messages':[HumanMessage(content="What is python")]}, config=config)
# print(response['messages'][-1].content)

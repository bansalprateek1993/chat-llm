from agentic_chatbot_rag_backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid
import tempfile
import os

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

if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False

add_thread(st.session_state["thread_id"])

st.sidebar.title("Previous Conversations")

st.sidebar.divider()
st.sidebar.subheader("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and not st.session_state["pdf_uploaded"]:
    with st.spinner("Indexing document..."):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            pdf_path = tmp.name

        ingest_rag_document(pdf_path)

        st.session_state["pdf_uploaded"] = True
        st.session_state["uploaded_pdf_name"] = uploaded_file.name

    st.sidebar.success(
        f"✅ {uploaded_file.name} indexed successfully"
    )
elif st.session_state["pdf_uploaded"]:
    st.sidebar.success(
        f"📄 Loaded: {st.session_state.get('uploaded_pdf_name')}"
    )

# Add sidebar
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

if st.sidebar.button("🗑️ Remove Uploaded Document"):
    st.session_state["pdf_uploaded"] = False
    st.session_state.pop("uploaded_pdf_name", None)

    if os.path.exists("faiss_db"):
        import shutil
        shutil.rmtree("faiss_db")

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

    with st.chat_message("assistant"):
        status = st.empty()
        response_placeholder = st.empty()

        final_response = ""

        for event in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="updates",
        ):

            # event is a dict like:
            # {'chat_node': {...}}
            # {'tools': {...}}

            if "chat_node" in event:
                status.info("🤔 Thinking...")

                messages = event["chat_node"].get("messages", [])
                if messages:
                    msg = messages[-1]

                    # Final AI response
                    if (
                        isinstance(msg, AIMessage)
                        and not msg.tool_calls
                        and msg.content
                    ):
                        final_response = msg.content
                        response_placeholder.markdown(final_response)

            elif "tools" in event:
                tool_messages = event["tools"].get("messages", [])

                for tool_msg in tool_messages:
                    tool_name = getattr(tool_msg, "name", "tool")
                    status.info(f"🔧 Running tool: `{tool_name}`")

        status.success("✅ Done")
        ai_message = final_response

    # from langchain_core.messages import AIMessage

    # response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]}, config=CONFIG)

    # ai_message = next(
    #     msg.content
    #     for msg in reversed(response["messages"])
    #     if isinstance(msg, AIMessage) and msg.content
    # )

    # Storing the AI message to message history.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # print(final_answer)
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    # with st.chat_message('assistant'):
    #     st.text(ai_message)


# st.write('User', user_input)

# thread_id = "1" # Unique id for users - different head id for the user
# config = {'configurable': {'thread_id': thread_id}}

# response = chatbot.invoke({'messages':[HumanMessage(content="What is python")]}, config=config)
# print(response['messages'][-1].content)
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import os
load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_KEY")
)


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    message = state['messages']
    response = llm.invoke(message)
    return {'messages': [response]}



conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    threads = checkpoint.list(None)
    all_threads = set()
    for thread in threads:
        all_threads.add(thread.config['configurable']['thread_id'])

    return list(all_threads)
# CONFIG = {"configurable":{"thread_id": "default_thread"}}
# res = chatbot.invoke(
#     {"messages": [HumanMessage(content="Hello, how is it going?")]},
#     config=CONFIG
# )

# print(res)

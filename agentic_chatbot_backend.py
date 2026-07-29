from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import math
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_KEY")
)

search_tool = TavilySearch(
    max_results=5,
    topic = "general",
    search_depth="advanced",
    tavily_api_key=os.getenv("TAVILY_API_KEY")  
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math caclulations.
    Input should be valid math expression.
    Example: 2+2, math.sqrt(16), 10*5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(expression, {"__builtin__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock proce for a given symbol (e.g. AAPL, TSLA)
    using Aplha vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=FGY6PWNXREZFY603"
    r = requests.get(url)
    return r.json()

# Make tool list
tools = [search_tool, calculator, get_stock_price]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    "LLM node that may answer or request a tool call"
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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

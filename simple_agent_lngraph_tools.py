import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")


llm_name = "gpt-3.5-turbo"

client = OpenAI(api_key=openai_key)
model = ChatOpenAI(api_key=openai_key, model=llm_name)


# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Below, implement a BasicToolNode that checks the most recent
# message in the state and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition

class BasicToolNode(ToolNode):
    """A node that runs the tools requested in the most recent message."""
    def __init__(self, tools):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self,inputs:dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found in state.")
        outputs = []
        for tool_call in message.tool_calls:
            tool_response = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(content=json.dumps(tool_response), name=tool_call["name"], tool_call_id=tool_call["id"]))
        return {"messages": outputs}


def bot(state: State):
    # print(state.items())
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}



# create tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
# rest = tool.invoke("What is the capital of France?")
# print(rest)



model_with_tools = model.bind_tools(tools)
# rest = model_with_tools.invoke("What's a 'node' in LangGraph?")
# print(rest)

# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
# tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph

from typing import Literal

# def route_tools(state: State) -> Literal["tools", "__end__"]:
#     """use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise , route to the end."""
#     if isinstance(state,list):
#         ai_messages = state[-1]
#     elif messages:= state.get("messages", []):
#         ai_messages = messages[-1]
#     else:
#         raise ValueError("No messages found in state.")
#     if hasattr(ai_messages, "tool_calls") and ai_messages.tool_calls:
#         return "tools"
#     return "__end__"

# graph_builder.add_conditional_edges(
#     "bot",
#     route_tools,
#     {"tools": "tools", "__end__": "__end__"},
# )
    

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

# STEP 5: Compile the graph
graph = graph_builder.compile(checkpointer=memory)
# MEMORY CODE CONTINUES ===
# Now we can run the chatbot and see how it behaves
# PICK A TRHEAD FIRST
config = {
    "configurable": {"thread_id": 1}
}  # a thread where the agent will dump its memory to
user_input = "Hi there! My name is Bond. and I have been happy for 100 years"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()


user_input = "do you remember my name, and how long have I been happy for?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()


snapshot = graph.get_state(config)
print(snapshot)


# from langchain_core.messages import BaseMessage

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": [("user", user_input)]}):
#         for value in event.values():
#             if isinstance(value["messages"][-1], BaseMessage):
#                 print("Assistant:", value["messages"][-1].content)

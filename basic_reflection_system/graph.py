from typing import Annotated, List
from typing_extensions import TypedDict
from chains import generation_chain, reflection_chain
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessageGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

REFLECT = "reflect"
GENERATE = "generate"
SHOULD_CONTINUE="should_continue"

def generate_node(state):
    """Node to generate a tweet based on the topic."""
    return generation_chain.invoke({
        "messages":state
    })

def reflect_node(state):
    """Node to reflect on the generated tweet."""
    response= reflection_chain.invoke({
        "messages": state
    })

    return HumanMessage(content=response.content)  # reflection is done via human 

def should_continue(state):
    """Condition to continue the reflection process."""
    if(len(state) > 4):
        return END
    return REFLECT  

# Create the graph
graph = MessageGraph()

# Add nodes
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Add edges
graph.add_edge(START, GENERATE)
graph.add_conditional_edges(GENERATE,should_continue)
graph.add_edge(REFLECT, GENERATE)

# Compile the graph
compiled_graph = graph.compile()

response=compiled_graph.invoke(HumanMessage(content="Write a tweet about AI Agents taking over content creation"))

# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# Printing each message with clear formatting
for idx, msg in enumerate(response):
    role = "🧑 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
    print(f"{'='*60}")
    print(f"{role} - Message {idx+1}")
    print(f"{'-'*60}")
    print(msg.content)
    print(f"{'-'*60}")
    if hasattr(msg, 'response_metadata') and msg.response_metadata:
        metadata = msg.response_metadata
        if 'token_usage' in metadata:
            usage = metadata['token_usage']
            print("📌 Token Usage:")
            print(f"    Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"    Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"    Total Tokens: {usage.get('total_tokens', 'N/A')}")
    print(f"{'='*60}\n")
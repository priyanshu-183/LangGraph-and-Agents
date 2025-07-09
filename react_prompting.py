from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Check if API keys are loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

tavily_search = TavilySearch(api_key=tavily_api_key)

tools = [
    Tool(
        name="tavily_search",
        func=tavily_search.invoke,
        description="Searches the web for real-time information or news."
    )
]

llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)

# Updated prompt with agent_scratchpad for proper ReAct functionality
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer user queries using the provided tools when necessary."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example query
user_input = "What's the latest news about Cricket icons at Wimbledon?"
result = agent_executor.invoke({"input": user_input})

print("\nAgent's Response:")
print(result["output"])
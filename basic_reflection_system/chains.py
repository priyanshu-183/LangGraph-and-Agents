from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

generation_prompt = ChatPromptTemplate.from_messages([
    (   
        "system", 
        "You're a Twitter Techie tasked with writing exciting twitter post."
        "Generate best twitter post possible for user request"
        "If user provides critique , respond with revised version of your previous attempts of tweet"
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a viral tweet infleuncer grading a tweet based on its engagement potential."
        "Generate a score between 1 and 10 for the tweet based on its creativity, engagement potential, and relevance to the topic."
        "Generate a critique and recommendations for improvement."
        "Always provide a score , detailed recommendation for virality, length and engagement potential."
    ),
    MessagesPlaceholder(variable_name="messages")
])

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
import os

load_dotenv()

# Azure Chat Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT") 
)

st.header('Research Tool')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt")
import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Feedback Sentiment Analyzer", page_icon="💬")

st.title("💬 Feedback Sentiment Analyzer")
st.write("Enter customer feedback and get an AI-generated response.")

# Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Output parser
parser = StrOutputParser()

# Pydantic Model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )

# Pydantic parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template="""
    Classify the sentiment of the following feedback text into positive or negative.

    Feedback:
    {feedback}

    {format_instruction}
    """,
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)

# Classification chain
classifier_chain = prompt1 | model | parser2

# Positive response prompt
prompt2 = PromptTemplate(
    template="""
    Write an appropriate professional response to this positive feedback:

    {feedback}
    """,
    input_variables=['feedback']
)

# Negative response prompt
prompt3 = PromptTemplate(
    template="""
    Write an appropriate professional response to this negative feedback:

    {feedback}
    """,
    input_variables=['feedback']
)

# Branching chain
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == 'positive',
        prompt2 | model | parser
    ),
    (
        lambda x: x.sentiment == 'negative',
        prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

# Final chain
chain = classifier_chain | branch_chain

# User input
feedback_input = st.text_area(
    "Enter Feedback",
    placeholder="Example: This phone is amazing!"
)

# Button
if st.button("Analyze Feedback"):

    if feedback_input.strip() == "":
        st.warning("Please enter feedback.")
    else:
        try:
            # Get sentiment first
            sentiment_result = classifier_chain.invoke({
                'feedback': feedback_input
            })

            # Get response
            response = chain.invoke({
                'feedback': feedback_input
            })

            # Display sentiment
            st.subheader("Detected Sentiment")
            st.success(sentiment_result.sentiment.capitalize())

            # Display AI response
            st.subheader("AI Response")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")
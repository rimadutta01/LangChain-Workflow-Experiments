import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)
from pydantic import BaseModel, Field
from typing import Literal
import os

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Feedback Classifier", page_icon="💬")

st.title("💬 Feedback Sentiment Classifier")
st.write("Enter feedback and get an AI-generated response.")

# Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Output Parsers
parser = StrOutputParser()

# Pydantic Model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for classification
prompt1 = PromptTemplate(
    template='''
Classify the sentiment of the following feedback text into positive or negative.

Feedback:
{feedback}

{format_instruction}
''',
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

# Positive Response Prompt
prompt2 = PromptTemplate(
    template='''
Write a polite response to this positive feedback:

{feedback}
''',
    input_variables=['feedback']
)

# Negative Response Prompt
prompt3 = PromptTemplate(
    template='''
Write a polite apology response to this negative feedback:

{feedback}
''',
    input_variables=['feedback']
)

# Branch Chain
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == 'positive',
        prompt2 | model | parser
    ),
    (
        lambda x: x.sentiment == 'negative',
        prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

# Final Chain
chain = classifier_chain | branch_chain

# User Input
user_feedback = st.text_area("Enter your feedback")

# Button
if st.button("Generate Response"):

    if user_feedback.strip() == "":
        st.warning("Please enter feedback.")
    else:
        with st.spinner("Analyzing feedback..."):

            result = chain.invoke({
                'feedback': user_feedback
            })

            st.success("Response Generated!")

            st.subheader("AI Response")
            st.write(result)
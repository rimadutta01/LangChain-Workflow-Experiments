from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="😊"
)

st.title("😊 Sentiment Classifier App")
st.write("Enter feedback and classify its sentiment.")

# Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Output Parser
parser = StrOutputParser()

# Prompt Template
prompt1 = PromptTemplate(
    template='''
Classify the sentiment of the following feedback text into only:
Positive or Negative.

Feedback:
{feedback}
''',
    input_variables=['feedback']
)

# Chain
classifier_chain = prompt1 | model | parser

# User Input
feedback = st.text_area("Enter Feedback")

# Button
if st.button("Classify Sentiment"):

    if feedback.strip() == "":
        st.warning("Please enter feedback.")
    else:
        with st.spinner("Analyzing sentiment..."):

            result = classifier_chain.invoke({
                'feedback': feedback
            })

            st.success("Classification Complete!")

            st.subheader("Sentiment")
            st.write(result)
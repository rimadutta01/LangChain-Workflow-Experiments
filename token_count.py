import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import tiktoken
import os

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Token Counter")

st.title("🔢 Azure OpenAI Token Counter")
st.write("Enter any text and check token usage.")


# Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)


# Token Counter Function
def count_tokens(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# User Input
user_input = st.text_area("Enter your text")

# Button
if st.button("Count Tokens"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        with st.spinner("Calculating tokens..."):

            # Count input tokens
            input_tokens = count_tokens(user_input)

             # Send to model
            response = model.invoke([
                HumanMessage(content=user_input)
            ])

            output_text = response.content

            # Count output tokens
            output_tokens = count_tokens(output_text)

            # Total tokens
            total_tokens = input_tokens + output_tokens

            # Display Results
            st.success("Token Count Generated!")

            st.subheader("Model Response")
            st.write(output_text)

            st.subheader("Token Usage")

            st.write(f"Input Tokens: {input_tokens}")
            st.write(f"Output Tokens: {output_tokens}")
            st.write(f"Total Tokens: {total_tokens}")




import streamlit as st
import pandas as pd
from transformers import pipeline

# Set your Hugging Face API token
HF_API_TOKEN = "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"

# Initialize the Hugging Face pipeline with the token
question_answering_pipeline = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad",
                                       use_auth_token=HF_API_TOKEN)


# Function to load CSV into DataFrame
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


# Function to ask the question-answering model a question about the DataFrame
def ask_question(question, context):
    result = question_answering_pipeline(question=question, context=context)
    return result


# Main Streamlit app
def main():
    st.title("CSV Content QA with Hugging Face")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Process CSV
        df = load_csv(uploaded_file)

        # Display the DataFrame in the app - this can be removed or modified
        st.dataframe(df)

        # Text input for asking a question
        question = st.text_input("Ask a question about the CSV data:")

        if question:
            # Assume the entire CSV is the context
            context = df.to_csv(index=False)

            # Get the answer from Hugging Face model
            answer = ask_question(question, context)
            st.write(answer['answer'])


# Run the app
if __name__ == "__main__":
    main()

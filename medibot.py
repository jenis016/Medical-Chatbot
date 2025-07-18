import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"
last_question = None  # Store last question globally

GREETING_RESPONSES = [
    "Hey there! 😊 How can I assist you today?",
    "Hello! 👋 Feel free to ask me anything.",
    "Hi! I'm here to help. What would you like to learn?",
]

UNKNOWN_RESPONSE = "I'm not sure what you mean 🤔. Could you clarify your question?"


@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store with embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt():
    """Ensure the chatbot answers the exact medical question asked."""
    return PromptTemplate(
        template="""
        Use the given context to answer the user's question **accurately and concisely**.
        If the user's query is too broad (like "Tell me about medical"), provide **a general structured overview** instead of random medical facts.

        **Context:** {context}  
        **User's Question:** {question}  

        **Guidelines for the response:**
        - Answer **only what is asked** (Do not give random medical terms).  
        - If the question is broad, **give a structured response** (e.g., "Medical knowledge includes fields like anatomy, pathology, and pharmacology").  
        - If the answer is not found, **say so** instead of making up an answer.  

        **Answer:**
        """,
        input_variables=["context", "question"]
    )


@st.cache_resource
def load_llm():
    """Load the Hugging Face language model."""
    HF_TOKEN = os.getenv("HF_TOKEN")
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 2048}
    )


def is_greeting(user_input):
    """Check if user input is a greeting."""
    greetings = ["hi", "hello", "hey", "heyy", "yo", "good morning", "good evening", "whats up"]
    return user_input.lower().strip() in greetings


def generate_response(question, word_limit=None):
    """Generate response for the given question, applying word limit if provided."""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "Error loading vectorstore."

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

        response = qa_chain.invoke({"query": question})
        result = response.get("result", UNKNOWN_RESPONSE)

        # Apply word limit if specified
        if word_limit:
            result_words = result.split()
            if len(result_words) > word_limit:
                result = " ".join(result_words[:word_limit]) + "..."

        return result

    except Exception as e:
        return f"An error occurred: {str(e)}"


def handle_user_input(user_input):
    """Handles user input: stores last question and processes word count input."""
    global last_question

    if user_input.strip().isdigit():  # If input is a number (word count)
        if last_question:
            return generate_response(last_question, word_limit=int(user_input))
        else:
            return "Please ask a question before specifying a word limit."

    last_question = user_input  # Store the question
    return generate_response(user_input)  # Generate response normally


def main():
    st.title("💬 AI Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your message here...")

    if prompt:
        st.chat_message("user").markdown(f"**You:** {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})

        if is_greeting(prompt):
            response = GREETING_RESPONSES[0]
        else:
            response = handle_user_input(prompt)

        formatted_result = f"**Bot:** {response}"
        st.chat_message("assistant").markdown(formatted_result, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
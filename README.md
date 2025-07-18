# Medical-Chatbot

An AI-powered medical chatbot built with Streamlit and LangChain, using state-of-the-art large language models and a custom medical knowledge base derived from PDFs (e.g., Gale Encyclopedia of Medicine). The chatbot can answer health-related queries using reliable content and provides concise, context-driven responses.

---

## Features

- **Natural Language Medical Q&A:** Type any medical question and get concise answers.
- **PDF-Based Knowledge:** Uses information extracted and embedded from reference PDFs (e.g., Gale Encyclopedia of Medicine).
- **Modern LLM Backend:** Relies on Hugging Face's Mistral-7B-Instruct model (or similar).
- **Search-Driven Contextual Answers:** Retrieves the most relevant information chunks for precise, evidence-based responses.
- **Flexible Streamlit UI:** Simple web chat interface for interactive Q&A.

---

## Setup Instructions

### 1. Clone the Repository

git clone <your-repo-url>
cd <repo-folder>


### 2. Python Environment

Requires Python 3.8+.

**Install all dependencies:**

pip install -r requirements.txt

(*requirements.txt* should include: `streamlit`, `langchain`, `faiss-cpu`, `huggingface_hub`, `sentence-transformers`, `pandas`, `openai`, and other dependencies.)

### 3. Download and Place PDFs

- Place your medical reference PDFs (e.g., Gale Encyclopedia of Medicine) into the `/data` folder for ingestion.

### 4. Embed PDF Knowledge Base

Run the script to process PDFs and build the FAISS vector database:

python main.py

This will:
- Split the PDFs into manageable text chunks.
- Generate sentence-transformer embeddings.
- Store them in `vectorstore/db_faiss`.

### 5. Hugging Face API Token

The chatbot requires access to the Hugging Face Inference API.  
Set your API token as an environment variable:

export HF_TOKEN=your_huggingface_token
(Or, add it to your environment or .env file.)

### 6. Run the Medical Chatbot

Launch the chatbot app:

streamlit run app.py

(If using the included example `medibot.py`, you may need to rename it to `app.py` or specify the script.)

The web UI will open in your browser. Type your medical question and get a response with trustworthy, PDF-sourced information.

---

## How It Works

- **PDF Ingestion:** The pipeline loads all PDFs in `/data`, splits them into overlapping chunks for context, and generates vector embeddings.
- **Vector Store:** All embeddings are stored in a FAISS database for efficient semantic search.
- **Retriever + LLM:** When a user asks a question, the chatbot retrieves the most relevant PDF chunks using vector search, provides them as context to the LLM (Mistral-7B-Instruct), and generates a precise answer.
- **Streamlit UI:** Clean, simple web-based chat interface.

---

## Customization & Extending

- Add more reliable medical PDFs to `/data` and re-run `main.py` to expand the bot's knowledge.
- Tune the prompt template in the code for your desired style or constraints.
- Swap out the LLM model with any supported Hugging Face endpoint by changing the `repo_id` in the code.

---

## Example Usage

Ask about conditions, symptoms, treatments, anatomy, or general medical facts, e.g.:
> *"What are the symptoms of diabetes?"*  
> *"Explain hypertension treatment."*  
> *"What does the Gale Encyclopedia say about asthma?"*

The chatbot will respond using verified context extracts from your PDF knowledge base.

---

## Security & Limitations

- This tool is **not a substitute for professional medical advice**. Always consult a healthcare provider for serious questions or emergencies.
- Answers are only as accurate as the source PDFs provided.

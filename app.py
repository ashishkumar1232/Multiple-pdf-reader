import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app UI
st.set_page_config(page_title="PDF Chat with Gemini", layout="wide")
st.title("💬 Chat with Multiple PDFs using Gemini Pro")

uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

user_question = st.text_input("Ask a question about your PDFs:")

# Read and extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Main logic
if uploaded_files and user_question:
    with st.spinner("Processing..."):
        # Step 1: Extract
        raw_text = extract_text_from_pdfs(uploaded_files)

        # Step 2: Chunk
        chunks = split_text(raw_text)

        # Step 3: Embed and Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        # Step 4: Retrieve Relevant Documents
        docs = vectorstore.similarity_search(user_question)

        # Step 5: Prompt and Chain
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        # Output
        st.subheader("📜 Answer:")
        st.write(response)

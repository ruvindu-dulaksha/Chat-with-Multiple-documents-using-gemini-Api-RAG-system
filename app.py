import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os.path

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key is missing. Please check your .env file.")
    raise ValueError("Google API Key not found in environment variables.")

# Configure Google Generative AI
import google.generativeai as genai
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("Error configuring Google Generative AI: " + str(e))
    raise

VECTOR_STORE_PATH = "faiss_index"

# Function to clean extracted text
def clean_text(raw_text):
    return " ".join(raw_text.split())

# Function to read text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                raw_text = page.extract_text() or ""
                text += clean_text(raw_text) + " "
        except Exception as e:
            st.warning(f"An error occurred while processing {pdf.name}: {str(e)}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create and save a vector store
def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
    except Exception as e:
        st.error("Error creating vector store: " + str(e))
        raise

# Function to load the vector store
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error("Error loading existing vector store: " + str(e))
            raise
    return None

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
       Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, 
       "The answer is not available in the context." Do not provide wrong answers.

       Context: {context}

       Question: {question}

       Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user questions
def user_input(user_question, vector_store):
    try:
        if vector_store is None:
            st.warning("No documents processed yet. Please upload PDF files first.")
            return
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error("Error processing the question: " + str(e))

# Main function
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini 💁")

    # Load existing vector store if available
    vector_store = load_vector_store()

    # User input for question
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, vector_store)

    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        create_vector_store(text_chunks)
                        st.success("Processing complete. You can now ask questions.")
                        vector_store = load_vector_store()  # Reload the updated vector store
                    else:
                        st.warning("No readable text found in the uploaded PDF files.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()

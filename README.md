# Chat with PDFs using Gemini

This project enables you to upload multiple PDF documents, process their contents into a vector store, and interactively ask questions about the uploaded content using Google Generative AI's Gemini model.

---

## Features

- **Upload Multiple PDFs**: Upload one or more PDF files.
- **Text Extraction and Cleaning**: Extracts and cleans text from PDFs to ensure high-quality context.
- **Text Splitting**: Splits extracted text into manageable chunks for efficient processing.
- **Vector Store**: Creates a vector store for semantic similarity search using FAISS and Google Generative AI embeddings.
- **Question Answering**: Ask detailed questions, and the app provides accurate answers based on the uploaded PDFs' content.
- **Interactive UI**: User-friendly Streamlit interface.

---

## Prerequisites

### 1. Tools and Libraries
Ensure you have the following installed:

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [LangChain](https://docs.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Generative AI API](https://developers.generativeai.google/)
- [dotenv](https://pypi.org/project/python-dotenv/)

### 2. API Key
Obtain your Google Generative AI API key and store it in a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ruvindu-dulaksha/Chat-with-Multiple-documents-using-gemini-Api-RAG-system.git
cd chat-with-pdfs
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your API Key

Create a `.env` file in the project root and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## Usage

### Upload PDF Files
- Use the sidebar to upload your PDF files.
- Click the **Submit & Process** button to extract text and create a vector store.

### Ask Questions
- Type your question in the main input box.
- The app will search for relevant content in the vector store and respond with detailed answers.

---

## Project Structure

```
.
├── app.py                 # Main application code
├── requirements.txt       # Required dependencies
├── .env                   # API key configuration file (not included in repo)
├── faiss_index/           # Directory to store vector store files
└── README.md              # Project documentation
```

---

## Key Functions

1. **`get_pdf_text(pdf_docs)`**
   - Extracts and cleans text from uploaded PDF files.

2. **`get_text_chunks(text)`**
   - Splits extracted text into smaller chunks for processing.

3. **`create_vector_store(text_chunks)`**
   - Creates and saves a FAISS-based vector store using Google Generative AI embeddings.

4. **`load_vector_store()`**
   - Loads an existing vector store if available.

5. **`get_conversational_chain()`**
   - Sets up a question-answering chain using the Gemini model.

6. **`user_input(user_question, vector_store)`**
   - Handles user queries and provides detailed answers.

---

## Notes

- Ensure the PDF files contain readable text. Scanned PDFs without OCR will not work.
- The app supports multiple PDF uploads but may experience latency for large files.

---

## Future Enhancements

- Add support for other file formats (e.g., DOCX, TXT).
- Implement document OCR for scanned PDFs.
- Improve error handling and logging.
- Optimize vector store creation for large datasets.

---

## License

This project is open-source and available under the [MIT License](LICENSE).


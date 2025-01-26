import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Streamlit UI
st.title("PDF Document Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# PDF Directory Upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Ensure that the uploaded files are stored in a directory
if uploaded_files:
    uploaded_files_path = "uploaded_files"
    if not os.path.exists(uploaded_files_path):
        os.makedirs(uploaded_files_path)

    # Save uploaded files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(uploaded_files_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"File '{uploaded_file.name}' uploaded successfully!")

    # Load PDFs into Documents
    loader = PyPDFDirectoryLoader(uploaded_files_path)
    docs = loader.load()

    # Ensure documents are loaded as Document objects and display their page content
    for doc in docs:
        st.write(f"Loaded document of type: {type(doc)}")  # Should show <class 'langchain.schema.Document'>

    # Store loaded documents in session state
    st.session_state.loaded_docs = docs
    st.write(f"Loaded {len(st.session_state.loaded_docs)} documents.")

    # Initialize Embeddings and Vector Store
    if not st.session_state.embeddings:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)

    # Create vector store using FAISS
    st.session_state.vectors = FAISS.from_documents(document_chunks, st.session_state.embeddings)

    # Initialize the LLM (ChatGroq)
    llm = ChatGroq(groq_api_key="your_groq_api_key", model_name="llama-3.3-70b-versatile", temperature=0)

    # Create ChatPrompt template for querying the documents
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert who answers questions based on the provided documents. Please answer based on the documents.

        <context>
        {context}
        </context>

        Question: {input}"""
    )

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Store document chain to session state
    st.session_state.retrieval_chain = document_chain

# Query input
query = st.text_input("Enter your query:")

# Process query and get answer
if st.button("Get Answer"):
    if query:
        # Use the FAISS vector store to retrieve relevant documents based on the query
        retriever = st.session_state.vectors.as_retriever(search_type="similarity", k=2)  # Retrieve top 2 relevant documents
        docs = retriever.get_relevant_documents(query)  # Perform similarity search with the query
        
        # If relevant documents are retrieved, prepare the context
        if docs:
            context = "\n".join([f"Document {i + 1}: Source: {doc.metadata.get('source', 'Unknown')}\n{doc.metadata.get('title', 'No Title')}\n{doc.page_content}" for i, doc in enumerate(docs)])

            # Use retrieval_chain to get the response
            response = st.session_state.retrieval_chain.invoke({"input": query, "context": context})

            # Check if the response has an 'answer' key and display it
            if isinstance(response, dict) and 'answer' in response:
                st.write("Answer: ", response['answer'])
            else:
                st.write("Could not retrieve answer.")
        else:
            st.write("No relevant documents found.")
    else:
        st.write("Please enter a query to get the answer.")

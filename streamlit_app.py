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
from langchain.schema import Document

# Streamlit UI
st.title("Document Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# PDF file uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Create a directory for storing uploaded PDFs
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")

    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Load the PDFs using PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader("uploaded_files")
    docs = loader.load()

    # Debugging: Print out the loaded docs to inspect their format
    st.write("Loaded documents:")
    st.write(docs)

    # Ensure docs are in the correct format (list of Document objects)
    if isinstance(docs, list):
        if isinstance(docs[0], str):  # If it's plain text
            docs = [Document(page_content=doc) for doc in docs]
        elif isinstance(docs[0], dict):  # If it's a dict format
            docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) for doc in docs]
    else:
        # If docs is not a list, wrap it manually
        docs = [Document(page_content=docs)]

    st.write(f"Documents wrapped into {len(docs)} Document objects.")

    # Store loaded documents in session state
    st.session_state.loaded_docs = docs

# LLM and Embedding initialization
llm = ChatGroq(groq_api_key="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", model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)

# Craft ChatPrompt Template
prompt = ChatPromptTemplate.from_template(
            """
            You are a Life Insurance specialist who needs to answer queries based on the information provided in the uploaded documents only. Please follow all the information in the documents, and answer as per the same.

            Do not answer anything except from the document information. Please do not skip any information from the document.

            Do not skip any information from the context. Answer appropriately as per the query asked.

            Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the documents, if asked.

            Generate tabular data wherever required to classify the difference between different parameters of policies.

            In the question when referred to as two companies, please understand that is for HDFC Life and Reliance Nippon.

            <context>
            {context}
            </context>

            Question: {input}"""
        )

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

# Split documents into smaller chunks
document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)
st.write(f"Number of chunks: {len(document_chunks)}")

# Stuff Document Chain Creation
document_chain = create_stuff_documents_chain(llm, prompt)

# Save document chain to session state
st.session_state.retrieval_chain = document_chain

# Query input from the user
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        # Directly pass the documents to the chain without using a retriever
        context = "\n".join([doc.page_content for doc in st.session_state.loaded_docs])
        response = st.session_state.retrieval_chain.invoke({"input": query, "context": context})

        # Check the structure of the response
        st.write("Response:")
        st.write(response)  # Print the entire response to inspect its format

        # If the response is a dictionary with 'answer' key, use it
        if isinstance(response, dict) and 'answer' in response:
            st.write(response['answer'])
        else:
            st.write("Here's your response!")
    else:
        st.write("No documents loaded. Please upload and process the PDFs first.")

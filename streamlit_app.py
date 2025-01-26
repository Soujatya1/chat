import os
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from io import BytesIO
import PyPDF2

# Streamlit UI
st.title("PDF Knowledge Repository")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# Directory to store uploaded files
uploaded_files_directory = "uploaded_files"
if not os.path.exists(uploaded_files_directory):
    os.makedirs(uploaded_files_directory)

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process PDFs automatically when uploaded
if uploaded_files:
    loader = PyPDFDirectoryLoader(uploaded_files_directory)
    docs = loader.load()
    
    # Debugging: Ensure that docs are loaded correctly
    st.write(f"Loaded {len(docs)} documents.")
    if len(docs) > 0:
        st.write(f"Content of the first document: {docs[0]['content']}")
    
    # Splitting the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    document_chunks = text_splitter.split_documents(docs)
    st.write(f"Number of chunks: {len(document_chunks)}")

    # Create the stuff documents chain
    llm = ChatGroq(
        groq_api_key="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",
        model_name='llama-3.3-70b-versatile',
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(
        """
        You are a specialist who needs to answer queries based on the information provided in the uploaded documents only. 

        Do not answer anything except from the information in the documents. Please do not skip any information from the tabular data in the documents.

        Do not skip any information from the context. Answer appropriately as per the query asked.

        Generate tabular data wherever required to classify differences or summarize content effectively.

        <context>
        {context}
        </context>

        Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retrieval_chain = document_chain

# Query and Response
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        if uploaded_files:
            # Make sure that context is correctly populated
            context = "\n".join([doc["content"] for doc in docs if isinstance(doc, dict)])
            st.write(f"Context being passed: {context[:500]}...")  # Debugging context

            # Invoke the chain with the context and query
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
            st.write("No documents loaded. Please upload and process PDF files first.")
    else:
        st.write("Please enter a query.")

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from io import BytesIO

# Streamlit UI
st.title("PDF Knowledge Repository")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

loaded_docs = st.session_state.loaded_docs

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Load and Process"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Use a file-like object (BytesIO) for PyPDFLoader
                pdf_file = BytesIO(uploaded_file.read())
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()

                # Debugging: Check the structure of the loaded documents
                if isinstance(docs, list):
                    for doc in docs:
                        if isinstance(doc, str):
                            st.write(f"Loaded document is a string: {doc[:200]}")  # Show a snippet of the text
                        elif hasattr(doc, "page_content"):
                            doc.metadata["source"] = uploaded_file.name
                        else:
                            st.write(f"Document has no 'page_content': {doc}")
                else:
                    st.write("Loaded content is not a list of documents.")
                
                # If documents are strings, wrap them in structured objects (add page_content)
                for doc in docs:
                    if isinstance(doc, str):
                        # Create a mock document structure with page_content
                        doc = {"page_content": doc}
                    loaded_docs.append(doc)
            except Exception as e:
                st.write(f"Error loading {uploaded_file.name}: {e}")
    else:
        st.write("Please upload at least one PDF file.")

    st.write(f"Loaded documents: {len(loaded_docs)}")

    # Store loaded documents in session state
    st.session_state.loaded_docs = loaded_docs

# LLM and Embedding initialization
llm = ChatGroq(
    groq_api_key="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",
    model_name='llama-3.3-70b-versatile',
    temperature=0
)

# Craft ChatPrompt Template
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

# Text Splitting
if st.session_state.loaded_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)
    st.write(f"Number of chunks: {len(document_chunks)}")

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Save document chain to session state
    st.session_state.retrieval_chain = document_chain

# Query and Response
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        if st.session_state.loaded_docs:
            # Directly pass the documents to the chain without using a retriever
            context = "\n".join([doc["page_content"] for doc in st.session_state.loaded_docs if isinstance(doc, dict)])
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

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
import PyPDF2
# Streamlit UI
st.title("PDF Knowledge Repository")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

loaded_docs = st.session_state.loaded_docs

#uploaded_file = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Load and Process PDF"):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Ensure that uploaded_file is a file-like object
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            total_pages = len(pdf_reader.pages)

            all_text = []
            
            # Loop through the pages of the PDF and extract text
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                if text:
                    all_text.append(text)

            # Process the extracted text
            st.write(f"Extracted {len(all_text)} pages of text from the PDF.")

            # Creating document structure for each page of text
            loaded_docs = []
            for page_num, text in enumerate(all_text):
                doc = {
                    "metadata": {
                        "source": uploaded_file.name,
                        "page_number": page_num + 1,
                    },
                    "content": text,
                }
                loaded_docs.append(doc)

            st.write(f"Loaded documents: {len(loaded_docs)}")

            # Optional: Displaying content of the first document
            st.write(f"Content of the first page: {loaded_docs[0]['content']}")

        except Exception as e:
            st.write(f"Error processing PDF: {e}")

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
if loaded_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    document_chunks = text_splitter.split_documents(loaded_docs)
    st.write(f"Number of chunks: {len(document_chunks)}")

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Save document chain to session state
    st.session_state.retrieval_chain = document_chain

# Query and Response
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        if loaded_docs:
            # Directly pass the documents to the chain without using a retriever
            context = "\n".join([doc["page_content"] for doc in loaded_docs if isinstance(doc, dict)])
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

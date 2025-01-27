import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Document GeN-ie")
st.subheader("Chat with your documents")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required .
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '.github/'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0)

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    # Prepare the context from documents
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    # Get the response from the chain
    response = chain.invoke({"question": question, "context": context})
    
    # Extract and return the content of the AIMessage response
    return response.content

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input("Ask a question:")

    if question:
        # Save the user's question to the conversation history
        st.session_state.conversation_history.append({"role": "user", "content": question})
        
        # Retrieve relevant documents
        related_documents = retrieve_docs(question)
        
        # Get the answer from the assistant
        answer = answer_question(question, related_documents)
        
        # Save the assistant's response to the conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})

    # Display the conversation history (user's question followed by the assistant's answer)
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

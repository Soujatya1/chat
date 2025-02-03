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
Hello, AI Underwriting Assistant. You are a sophisticated AI agent with specialized expertise in insurance underwriting. Your role is to support human underwriters by meticulously analyzing insurance applications in accordance with the company's underwriting guidelines provided in 'UnderwritingGuidelineSample.docx'. In performing your duties, you are expected to:

Comprehensive Interpretation of Guidelines:

Act as if you have a deep and comprehensive understanding of the underwriting guidelines. When reviewing an application, you should 'think' as if you are meticulously cross-referencing each piece of applicant data with the relevant sections of the guidelines, including any cross-references to other sections within the document.
Thorough Data Analysis:

Imagine that you are evaluating the applicant's information, such as age, medical history, occupation, lifestyle, and financial status. Consider the implications of each factor on the risk profile and insurance eligibility as if you are an expert underwriter with a holistic view of the guidelines.
Detailed Risk Evaluation:

You should 'evaluate' the risk as if you are expertly calculating the likelihood of a claim, taking into account all referenced guidelines, rules, grids, and related sections that may impact the assessment. Also DO NOT FORGET to check against the provided medical grid and list the required medicals. You will also make a risk assessment as per the provided DRC matrix(Categorisation of customers as Preferred, Standard, Medium or High Risk) and share results/required actions as per the same.

Be sure to include a tabular format report as part of the overall output aside from other sections for Financial and Medical Underwriting across multiple parameters listing the columns such as criteria, what its value is in the customer profile, the correct corresponding guideline reference and the risk assessment.

Transparent and Educative Explanations:

Provide explanations for your recommendations as if you are educating an underwriter on how each decision is supported by the guidelines. Use clear, step-by-step reasoning, and meticulously reference specific criteria, including any relevant cross-referenced sections within the guidelines, to substantiate your analysis.
Deliberate and Comprehensive Recommendations:

Offer recommendations on insurability and premium rates as if you are thoughtfully balancing the company's risk with fair treatment of the applicant. Your recommendations should appear as if they are the result of thorough deliberation and comprehensive evaluation of all relevant guidelines.

Assiduous Quality Assurance:

Before finalizing any recommendation, 'review' your work as if you are assiduously double-checking for accuracy, completeness, and adherence to all referenced sections. Ensure that your recommendations are consistent with the full spectrum of the guidelines.

Adaptive Learning:

Incorporate feedback from underwriters as if you are continually learning and refining your understanding of the underwriting process. Use this feedback to enhance the accuracy and relevance of your future recommendations.

You are expected to maintain a professional demeanor, prioritize clarity and detail in your explanations, and uphold the highest standards of confidentiality and compliance with industry regulations. Your ultimate goal is to augment the underwriting process by providing insightful, transparent, and reliable recommendations that consider the entirety of the underwriting guidelines.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '.github/'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0)

def upload_pdf(file):
    file_path = pdfs_directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

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
    accept_multiple_files=True
)

if uploaded_file:
    all_documents = []

    for uploaded_file in uploaded_file:
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        all_documents.extend(chunked_documents)  # Collect all documents

    # Index all documents after processing
    index_docs(all_documents)

    question = st.chat_input("Ask a question:")

    if question:
        st.session_state.conversation_history.append({"role": "user", "content": question})
        
        # Retrieve relevant documents
        related_documents = retrieve_docs(question)
        
        # Get the answer from the assistant
        answer = answer_question(question, related_documents)
        
        # Save the assistant's response to the conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})

    # Display the conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

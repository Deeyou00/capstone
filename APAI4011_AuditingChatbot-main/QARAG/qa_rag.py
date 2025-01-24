import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

<<<<<<< HEAD
from dotenv import load_dotenv
import os
load_dotenv()
# Load OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

=======
>>>>>>> 2ffd424 (Initial commit - reset history)
# Step 1: Chunk PDF into documents
def pdf_to_documents(pdf_path, chunk_size=500, chunk_overlap=50):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# Step 2: Create a vector store for retrieval
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Step 3: Retrieve relevant documents
def retrieve_documents(vectorstore, query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs

# Step 4: Use GPT to process the query and retrieved documents
def process_with_gpt(question, relevant_docs):
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
<<<<<<< HEAD
    print(context, "?"*100)
=======
>>>>>>> 2ffd424 (Initial commit - reset history)
    prompt = (
        f"You are a professional information extractor and experience auditor. Use the following context to answer the question:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
    messages = [
<<<<<<< HEAD
        {"role": "system", "content": "You are a professional information extraction"},   
        {"role": "user", "content": f"{prompt}"}
    ]

    response = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages, 
    )
    
    # Extract the GPT response
    response = response.choices[0].message.content
    return response
=======
        {"role": "system", "content": "You are a professional information extraction assistant."},
        {"role": "user", "content": f"{prompt}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
    )
    return response['choices'][0]['message']['content']
>>>>>>> 2ffd424 (Initial commit - reset history)

# Main RAG Workflow
def rag_workflow(pdf_path, question):
    # Chunk the PDF
    documents = pdf_to_documents(pdf_path)
    
    # Create vector store
    vectorstore = create_vector_store(documents)
    
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(vectorstore, question)
    
    # Process with GPT
    answer = process_with_gpt(question, relevant_docs)
    
    return answer

def qa_rag_run(pdf_path = "Data/document.pdf", question = "Justin is handsome?"):
    result = rag_workflow(pdf_path, question)
<<<<<<< HEAD
    return result

"""answer = qa_rag_run(pdf_path = "Data/uploaded.pdf", question = "what is stat7101")
print(answer)"""
=======
    return result
>>>>>>> 2ffd424 (Initial commit - reset history)

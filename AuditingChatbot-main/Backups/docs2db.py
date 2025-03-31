from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import sqlite3
import os

def view_table_structure(db_path, table_name,flag = False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conn.text_factory = lambda x: str(x, 'latin1')

    # Retrieve table structure information
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = cursor.fetchall()
    print(f"Table structure for {table_name}:")
    for column in table_info:
        print(column)
    if flag == True:
        # Retrieve and print the first 3 rows of data
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        num_rows = len(rows)
        num_columns = len(table_info)

        print("\nDatabase size:")
        print(f"{num_rows} rows x {num_columns} columns")

    conn.close()

def view_db_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to retrieve the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:")
    for table in tables:
        print(table[0])

    conn.close()
    return tables

def load_existing_retriever_from_db(db_path, collection_name="rag_milvus"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Assuming your retriever data is stored in a table called 'retriever_data'
    cursor.execute(f"SELECT * FROM {collection_name}")
    retriever_data = cursor.fetchall()

    conn.close()

    return retriever_data

def get_retriver_from_db(doc_splits, db_path="/root/users/jusjus/Self/audit_rag.db", collection_name="rag_milvus"):
    # Add to Milvus
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Milvus.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=hf,
        connection_args={"uri": db_path},
        #existing_retriever=existing_retriever
    )
    retriever = vectorstore.as_retriever()
    return retriever

def process_txt_file_to_db(file_path, existing_retriever=None):
    with open(file_path, 'r') as file:
        full_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_splits = text_splitter.split_text(full_text)
    doc_splits = [Document(page_content=i, metadata={"source":file_path, "page":idx}) for idx, i in enumerate(doc_splits)]
    return doc_splits

def process_pdf_file_to_db(file_path, existing_retriever=None):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def process_web_url_to_db(urls, existing_retriever=None):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_token_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def aggregate_doc_splits(doc_splits_list):
    aggregated_doc_splits = []
    for doc_splits in doc_splits_list:
        aggregated_doc_splits.extend(doc_splits)
    return aggregated_doc_splits

def process_files_in_folder(folder_path):
    doc_splits_list = []

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            if item.endswith(".txt"):
                doc_splits_list.append(process_txt_file_to_db(item_path))
            elif item.endswith(".pdf"):
                doc_splits_list.append(process_pdf_file_to_db(item_path))
            else:
                doc_splits_list.append(process_web_url_to_db([item_path]))

    return aggregate_doc_splits(doc_splits_list)


def store_as_db(db_path = "/root/users/jusjus/Self/audit_rag.db"):
    view_db_tables(db_path)
    view_table_structure(db_path, "rag_milvus")
    #existing_retriever = load_existing_retriever_from_db(db_path)  # Load your existing retriever

    # Example usage
    folder_path = "/root/users/jusjus/Self/LLaMA-Factory/data/zips/auditing_books_txt"
    doc_splits = process_files_in_folder(folder_path)

    #input()
    # Get retriever from aggregated doc_splits
    retriever = get_retriver_from_db(doc_splits)

    view_table_structure(db_path, "collection_meta")
    view_table_structure(db_path, "rag_milvus", True)




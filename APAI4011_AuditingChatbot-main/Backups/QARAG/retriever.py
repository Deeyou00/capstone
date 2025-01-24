from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

URI = "./milvus_example.db"

# Setup retriever
embeddings = HuggingFaceEmbeddings()
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)
retriever = vector_store.as_retriever()

def retrieve_documents(question):
    return retriever.get_relevant_documents(question)

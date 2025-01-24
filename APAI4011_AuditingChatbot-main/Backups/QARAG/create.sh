#!/bin/bash

# Define the root directory
ROOT_DIR="./"
FILES=("app.py" "config.py" "retriever.py" "grader.py" "generator.py" "router.py" "workflow.py" "requirements.txt")

# Create files with initial content
for FILE in "${FILES[@]}"; do
    echo "Creating $ROOT_DIR$FILE..."
    FILE_PATH="$ROOT_DIR$FILE"
    touch "$FILE_PATH"
    
    # Add boilerplate to each file
    case $FILE in
        "app.py")
            cat << 'EOF' > "$FILE_PATH"
from config import llm
from retriever import retrieve_documents
from grader import grade_relevance, grade_generation_support
from generator import generate_answer
from workflow import build_workflow

if __name__ == "__main__":
    # Initialize workflow
    workflow = build_workflow(
        retriever=retrieve_documents,
        generator=generate_answer,
        grader=grade_relevance,
    )

    # Example query
    query = {"question": "Did Emmanuel Macron visit Germany recently?"}
    results = workflow.run(query)
    print(results["generation"])
EOF
            ;;
        "config.py")
            cat << 'EOF' > "$FILE_PATH"
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# Load OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0)
EOF
            ;;
        "retriever.py")
            cat << 'EOF' > "$FILE_PATH"
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
EOF
            ;;
        "grader.py")
            cat << 'EOF' > "$FILE_PATH"
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def create_grader_prompt(template, input_variables):
    return PromptTemplate(template=template, input_variables=input_variables)

def grade_relevance(llm, question, document):
    prompt = create_grader_prompt(
        template="You are a grader assessing relevance of a document... Question: {question}, Document: {document}",
        input_variables=["question", "document"],
    )
    grader = prompt | llm | JsonOutputParser()
    return grader.invoke({"question": question, "document": document})

def grade_generation_support(llm, documents, generation):
    prompt = create_grader_prompt(
        template="Assess if the generation is grounded in documents... Documents: {documents}, Generation: {generation}",
        input_variables=["documents", "generation"],
    )
    grader = prompt | llm | JsonOutputParser()
    return grader.invoke({"documents": documents, "generation": generation})
EOF
            ;;
        "generator.py")
            cat << 'EOF' > "$FILE_PATH"
from langchain.prompts import PromptTemplate

def generate_answer(llm, question, context):
    prompt = PromptTemplate(
        template="You are a question-answering assistant... Question: {question}, Context: {context}",
        input_variables=["question", "context"],
    )
    return prompt | llm | StrOutputParser()
EOF
            ;;
        "router.py")
            cat << 'EOF' > "$FILE_PATH"
from langchain.prompts import PromptTemplate

def route_question(llm, question):
    prompt = PromptTemplate(
        template="Route the question to web_search or vectorstore... Question: {question}",
        input_variables=["question"],
    )
    router = prompt | llm | JsonOutputParser()
    return router.invoke({"question": question})
EOF
            ;;
        "workflow.py")
            cat << 'EOF' > "$FILE_PATH"
from langgraph.graph import StateGraph, END

def build_workflow(retriever, generator, grader):
    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retriever.retrieve_documents)
    workflow.add_node("grade_documents", grader.grade_relevance)
    workflow.add_node("generate", generator.generate_answer)

    workflow.set_conditional_entry_point(
        router.route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    return workflow
EOF
            ;;
        "requirements.txt")
            cat << 'EOF' > "$FILE_PATH"
langchain
langgraph
python-dotenv
EOF
            ;;
    esac
done

echo "All files and structure created!"
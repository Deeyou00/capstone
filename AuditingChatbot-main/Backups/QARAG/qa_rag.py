from config import llm
from retriever import retrieve_documents
from grader import grade_relevance, grade_generation_support
from generator import generate_answer
from workflow import build_workflow

def rag_workflow(question):
    workflow = build_workflow(
        retriever=retrieve_documents,
        generator=generate_answer,
        grader=grade_relevance,
    )

    # Example query
    query = {"question": question}
    results = workflow.run(query)
    return results

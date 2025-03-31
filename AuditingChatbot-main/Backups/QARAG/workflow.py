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

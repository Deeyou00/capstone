from langchain.prompts import PromptTemplate

def route_question(llm, question):
    prompt = PromptTemplate(
        template="Route the question to web_search or vectorstore... Question: {question}",
        input_variables=["question"],
    )
    router = prompt | llm | JsonOutputParser()
    return router.invoke({"question": question})

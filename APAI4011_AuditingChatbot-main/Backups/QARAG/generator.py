from langchain.prompts import PromptTemplate

def generate_answer(llm, question, context):
    prompt = PromptTemplate(
        template="You are a question-answering assistant... Question: {question}, Context: {context}",
        input_variables=["question", "context"],
    )
    return prompt | llm | StrOutputParser()

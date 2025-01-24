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

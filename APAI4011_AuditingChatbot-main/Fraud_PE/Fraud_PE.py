import openai
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from tqdm import tqdm


# Load OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def Fraud_init_eval(financial_report_section):
    prompt_content = f"""
    You are an expert financial analyst and writing evaluator. You will analyze a provided section of a financial report and evaluate the financial data, writing style, and likelihood of fraudulent practices. Your evaluation should be supported by numerical evidence and quoted sentences from the text.
    Here is the task:  
    1. Determine if the financial data is normal for the company in the industry for the given year.  
    2. Identify any sentences that are over-portraying the company with a lack of objectivity (e.g., excessive positive bias or unverifiable claims).  
    3. Assess whether the company is likely to be involved in fraudulent practices, citing specific red flags if any are observed.

    **Instructions:**  
    - Use evidence in numbers or quotes to support your conclusions.  
    - Compare the financial data with general industry benchmarks when applicable.  
    - Be concise and clear in your responses.  

    **Input:**  
    - A section of a financial report:  
    "{financial_report_section}"  

    **Output:**  
    1. **Industry Comparison:**  
    [Your assessment here, with supporting evidence.]

    2. **Writing Objectivity:**  
    [Your analysis of overly biased sentences, with quoted examples.]

    3. **Fraud Likelihood:**  
    [Your assessment of potential fraud, with red flags and evidence.]
    """
    
    messages = [
            {"role": "system", "content": "You are a professional information extraction"},   
            {"role": "user", "content": f"{prompt_content}"}
        ]

    response = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        
    )

    response = response.choices[0].message.content
    return response

def Fraud_aggr_eval(list_init_eval):
    """
    Aggregates initial evaluations into a single paragraph, performs prompt engineering,
    and answers three key questions:
    1. Is the company likely to be a fraud case?
    2. What are the evidence for/against the fraud statement?
    3. Provide confidence in the result (0-10).
    
    Args:
        list_init_eval (list of str): A list of initial evaluations.
    
    Returns:
        dict: Contains answers to the three questions.
    """
    # Aggregate evaluations into a single paragraph
    aggregated_eval = " ".join(list_init_eval)
    
    # Define the prompt
    prompt = f"""
    Given the following aggregated evaluation of a company's financial report:
    
    {aggregated_eval}
    
    Please analyze the content and answer the following questions:
    1. Is the company likely to be a fraud case? Provide a 'Yes' or 'No' answer.
    2. What are the evidence for or against the fraud statement? Use quoted sentences and numeric evidence where possible.
    3. Provide a confidence score for the result on a scale of 0-10.
    """
    
    # Send the prompt to GPT
    messages = [
        {"role": "system", "content": "You are a professional information extraction"},   
        {"role": "user", "content": f"{prompt}"}
    ]

    response = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages, 
    )
    
    # Extract the GPT response
    response = response.choices[0].message.content
    
    # Optionally, parse the output if required
    return {
        "aggregated_eval": aggregated_eval,
        "analysis": response
    }
    
def res_from_demo_data(csv_path):
    """
    Processes a CSV or Excel file and evaluates fraud likelihood for each row using LLM.
    Adds a new column 'LLM_output' with the GPT analysis results.

    Args:
        csv_path (str): Path to the CSV or Excel file.

    Returns:
        pd.DataFrame: DataFrame with an additional 'LLM_output' column.
    """
    # Load data
    if csv_path.endswith(".csv"):
        df = pd.read_csv(csv_path)
    elif csv_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(csv_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    # Create a copy of the DataFrame
    df_res = df.copy()
    
    # Initialize an empty column for LLM output
    df_res["LLM_output"] = None
    
    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Extract relevant columns into a list for Fraud_init_eval
        list_init_eval = []
        
        # Process each column value for Fraud_init_eval
        for col in row:
            init_eval = Fraud_init_eval(str(col))  # Convert column value to string and pass to Fraud_init_eval
            if init_eval:  # If Fraud_init_eval returns a result
                list_init_eval.append(init_eval)
        
        # Run Fraud_aggr_eval on the collected initial evaluations
        result = Fraud_aggr_eval(list_init_eval)
        
        # Save the result to the new column
        if result:
            df_res.at[index, "LLM_output"] = result["analysis"]
        else:
            df_res.at[index, "LLM_output"] = "Error or no result"
    
    return df_res

# Example usage
# Assuming Fraud_init_eval and Fraud_aggr_eval are defined
csv_path = "/root/users/jusjus/Self/APAI4011_AuditingChatbot/Data/modified_final_dataset.csv"
df_with_output = res_from_demo_data(csv_path)

df_with_output.to_csv("/root/users/jusjus/Self/APAI4011_AuditingChatbot/Data/modified_final_dataset_llm.csv", index=False)
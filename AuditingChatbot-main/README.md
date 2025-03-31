# AuditingChatbot

# Finetune:
	Qwen-2-7B-QA3000 (Baidu)
	Gemma-2-9B-QA3000 (GOOGLE)

# RAG Database
	3000 QA -> Finetune
	20 auditing textbook -> RAG anwser question

	200-300 Financial Report 	-> Query
	  		 		-> Fraud

# RAG

<img src="https://github.com/Justinfungi/APAI4011_AuditingChatbot/blob/main/statics/rag.png"></img>  	

# Fraud Detection
	Prepare 5 key item + LLM output
	Multiple model
	Majority Vote
	  		 
# WebAPP

uvicorn llm_api:app --host 0.0.0.0 --port 4567

curl -X POST http://localhost:4567/generate/ -H "Content-Type: application/json" -d "{\"text\": \"What is the significance of audit committees in corporate governance?\"}"


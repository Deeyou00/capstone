uvicorn llm_api:app --host 0.0.0.0 --port 4567

<<<<<<< HEAD
curl -X POST http://localhost:4567/generate/ -H "Content-Type: application/json" -d "{\"text\": \"What is the significance of audit committees in corporate governance?\"}"

=======
curl -X POST http://localhost:4567/generate/ -H "Content-Type: application/json" -d "{\"text\": \"What is the significance of audit committees in corporate governance?\"}"
>>>>>>> 2ffd424 (Initial commit - reset history)

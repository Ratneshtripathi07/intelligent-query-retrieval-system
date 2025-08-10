import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Import only one function from core.processing
from core.processing import process_document_and_questions

load_dotenv()
app = FastAPI(title="Intelligent Query-Retrieval System")
security = HTTPBearer()
EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid auth token")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query_pipeline(request: QueryRequest, token: str = Security(verify_token)):
    # Call our main processing function directly
    # This will be a bit slow the first time, but will run smoothly on your powerful PC
    answers = await process_document_and_questions(request.documents, request.questions)
    return QueryResponse(answers=answers)

@app.get("/")
async def read_root():
    return {"status": "ok"}
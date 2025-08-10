# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
#                                           V1.4 all-MiniLM-L6-v2
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
import os
import tempfile
import hashlib
import httpx
import asyncio
import time
import logging
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
# NEW: Import LangChain's official Embeddings class
from langchain_core.embeddings import Embeddings

load_dotenv()

INDEX_DIR = "persistent_indexes"
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Corrected Local Embeddings Client ---
class LocalEmbeddings(Embeddings):
    # def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
    # whenever switching models of different v dimensions, delete the older indexes in the persisten index folder
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading local Sentence Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Local model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

embeddings_client = LocalEmbeddings()
llm_client = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", temperature=0)

async def process_document_and_questions(doc_url: str, questions: List[str]) -> List[str]:
    logger.info(f"Processing request for document: {doc_url}")
    
    try:
        url_hash = hashlib.sha256(doc_url.encode()).hexdigest()
        index_path = os.path.join(INDEX_DIR, f"faiss_local_{url_hash}")
        
        vector_store = None
        
        if os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from: {index_path}")
            loop = asyncio.get_running_loop()
            vector_store = await loop.run_in_executor(
                None, lambda: FAISS.load_local(index_path, embeddings_client, allow_dangerous_deserialization=True)
            )
        else:
            logger.info(f"No existing index found. Creating a new one...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(doc_url, follow_redirects=True)
                response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                loader = PyMuPDFLoader(tmp_file_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_documents(documents)
                
                logger.info(f"Building FAISS index for {len(chunks)} chunks using local model...")
                loop = asyncio.get_running_loop()
                vector_store = await loop.run_in_executor(
                    None, lambda: FAISS.from_documents(chunks, embeddings_client)
                )
                
                logger.info(f"Saving new index to disk: {index_path}")
                await loop.run_in_executor(None, lambda: vector_store.save_local(index_path))
            finally:
                os.remove(tmp_file_path)

        # --- RAG Setup ---
        retriever = vector_store.as_retriever(search_kwargs={"k": 7})

        # --- PROMPT FIX START ---
        # Here the correct, detailed prompt has been added
        prompt_template = """
        You are an expert AI data extractor. Answer the question with extreme precision based ONLY on the provided context.

        INSTRUCTIONS:
        - Provide the answer in bullet points if there are multiple conditions, steps, or parts.
        - If the answer is a single, direct value (like a number or a short phrase), provide only that value.
        - Do not add any introductory phrases. Be direct and concise.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        PRECISE ANSWER:
        """
        # --- PROMPT FIX END ---
        
        prompt = PromptTemplate.from_template(prompt_template)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm_client | StrOutputParser())

        # --- Process Questions Concurrently ---
        async def process_single_question(question: str) -> str:
            try:
                answer = await rag_chain.ainvoke(question)
                return answer.strip()
            except Exception as e:
                logger.error(f"Error for question '{question[:50]}...': {e}")
                return "Error processing this question."

        tasks = [process_single_question(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        
        return answers

    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
        return [f"A critical error occurred: {e}"] * len(questions)
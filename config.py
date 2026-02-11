import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    
    SIMILARITY_THRESHOLD = 0.3
    TOP_K = 3

settings = Config()

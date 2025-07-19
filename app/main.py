from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.inference import generate_stream
import os

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat_stream(req: ChatRequest):
    def token_streamer():
        for token in generate_stream(req.prompt):
            yield token
    return StreamingResponse(token_streamer(), media_type="text/plain")

# Mount static files *after* routes so API endpoints work
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
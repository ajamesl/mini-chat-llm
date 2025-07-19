"""Main FastAPI application for the mini-chat-llm service."""
import os
from typing import Generator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import generate_stream

app = FastAPI()


class ChatRequest(BaseModel):
    """Request model for chat endpoint containing the user prompt."""
    prompt: str


@app.post("/chat")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Endpoint to stream chat responses from the language model.

    Args:
        req (ChatRequest): The chat request containing the user prompt.

    Returns:
        StreamingResponse: A streaming response with generated text.
    """
    def token_streamer() -> Generator[str, None, None]:
        """Generator that yields tokens from the language model stream."""
        for token in generate_stream(req.prompt):
            yield token

    return StreamingResponse(token_streamer(), media_type="text/plain")


app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static",
)

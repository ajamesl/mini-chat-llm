"""Main FastAPI application for the MiniChat service."""
import os
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.inference import generate_stream

app = FastAPI()

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Mount static files from the root static directory
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "templates"))


class ChatRequest(BaseModel):
    """Request model for chat endpoint containing the user prompt."""
    prompt: str


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test-static")
def test_static():
    """Test endpoint to verify static files are being served."""
    import os
    static_path = os.path.join(PROJECT_ROOT, "static", "speech_bubble.png")
    return {
        "static_path": static_path,
        "file_exists": os.path.exists(static_path),
        "file_size": os.path.getsize(static_path) if os.path.exists(static_path) else None
    }


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

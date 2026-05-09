# 💬 MiniChat: A Multilingual LLM Assistant

A FastAPI web application serving a multilingual chat assistant, powered by a Qwen3 0.6B Base model. The model is first fine-tuned with supervised instruction data (SFT) and then further optimised using Reinforcement Learning from Human Feedback (RLHF) via Proximal Policy Optimization (PPO).

## Features

- **Multilingual Chat**: Supports conversational AI in multiple languages
- **Modern Web Interface**: Clean, responsive UI with real-time streaming
- **Token-by-Token Streaming**: Watch responses generate live
- **Advanced Model**: Qwen3 0.6B Base, SFT + RLHF PPO
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **FastAPI Backend**: RESTful API with automatic documentation

## Project Structure

```
mini-chat-llm/
├── src/                    # Training scripts and utilities
│   ├── __init__.py
│   ├── config.py           # Training configuration and hyperparameters
│   ├── utils.py            # Shared utility functions (e.g., seed, data prep)
│   ├── model.py            # Model and tokenizer loading helpers
│   ├── sft_train.py        # Supervised fine-tuning (SFT) script
│   └── ppo_train.py        # RLHF PPO training script
├── app/                   # FastAPI application
│   ├── __init__.py
│   ├── main.py             # FastAPI app and routes
│   └── inference.py        # Model loading and text generation
│   └── static/
│       └── index.html      # Web UI
├── checkpoints/           # Model checkpoints
│   ├── sft_model/         # SFT checkpoint files and final model
│   ├── ppo_model/         # PPO checkpoint files and final model
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Python dependencies
├── uv.lock                # Dependency lock file
└── README.md              # This file
```

## 🛠️ Installation

### Option 1: Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd mini-chat-llm
   ```
2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

### Option 2: Using uv (Development)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd mini-chat-llm
   ```
2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Install dependencies**:
   ```bash
   uv sync
   ```
4. **Run the application**:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 3: Using pip (Development)

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd mini-chat-llm
   ```
2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Web Application

1. **Start the server**:
   ```bash
   docker-compose up
   ```
2. **Open your browser** and navigate to `http://localhost:8000`
3. **Chat with the assistant**: Enter your prompt and receive multilingual responses in real time.

### API Usage

#### Chat Endpoint (POST)
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello!"}'
```
This endpoint returns a streaming response with the assistant's reply, token by token.

#### Health Check
(If implemented, add details here)

## Model Architecture & Training

- **Base Model**: Qwen3 0.6B (multilingual, open weights)
- **Supervised Fine-Tuning (SFT)**: Trained on instruction-following data (e.g., OpenAssistant/oasst1)
- **RLHF PPO**: Further optimized using human feedback and Proximal Policy Optimization (PPO) with a reward model
- **Adapters**: LoRA adapters used for efficient fine-tuning
- **Final Model**: Merged for direct inference
- **Modular Training Code**: Training logic is now split into `config.py` (hyperparameters), `utils.py` (helpers), `model.py` (model/tokenizer loading), and the main scripts (`sft_train.py`, `ppo_train.py`).

### Training Pipeline

1. **Supervised Fine-Tuning (SFT)**
   - Script: `src/sft_train.py`
   - Config: `src/config.py` (edit hyperparameters here)
   - Utilities: `src/utils.py` (data prep, seeding)
   - Model loading: `src/model.py`
   - Data: Instruction-response pairs (e.g., OpenAssistant/oasst1)
   - LoRA adapters applied for parameter-efficient training
   - Model checkpoint saved to `checkpoints/sft_model/`
   - **Run:**
     ```bash
     uv run src/sft_train.py
     ```

2. **RLHF with PPO**
   - Script: `src/ppo_train.py`
   - Config: `src/config.py` (edit hyperparameters here)
   - Utilities: `src/utils.py`
   - Model loading: `src/model.py`
   - Reward model: e.g., Skywork/Skywork-Reward-V2-Qwen3-0.6B
   - PPO optimization using TRL library
   - Model checkpoint saved to `checkpoints/ppo_model/`
   - **Run:**
     ```bash
     uv run src/ppo_train.py
     ```

3. **Merging for Inference**
   - Final model merged and saved to `checkpoints/ppo_model/` for direct loading by the FastAPI app

## Deployment

### Local Development
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production (Docker)
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation powered by FastAPI.

## UI Features

- **Modern Design**: Clean, responsive interface
- **Real-time Streaming**: Watch responses generate token by token
- **Error Handling**: Graceful error messages and loading states
- **Responsive Layout**: Works on desktop and mobile devices

## 🐳 Docker Support

The project includes full Docker support:
- `Dockerfile`: Multi-stage build with Python 3.12
- `docker-compose.yml`: Easy development setup
- `.dockerignore`: Optimized build context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen3 model by Alibaba
- RLHF and PPO pipeline inspired by TRL and OpenAssistant
- Built with FastAPI and modern web technologies

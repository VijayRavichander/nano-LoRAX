# FastAPI LoRA Adapter Proxy

## Overview

This project provides a FastAPI-based server that acts as a proxy to dynamically download, load, and unload LoRA (Low-Rank Adaptation) adapters based on user requests.&#x20;

## Features

- **Dynamic LoRA Management**: Load and unload LoRA adapters on demand.
- **Proxy Server**: Acts as a middleware to facilitate requests for different LoRA adapters.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/VijayRavichander/nano-LoRAX
   ```

2. Create a virtual environment and activate it:

   ```sh
    wget -qO- https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv venv
    source .venv/bin/activate
   ```

3. Install dependencies:

   ```sh
    uv pip install -r requirements.txt
   ```

## Usage

### Start the FastAPI Server

Export Variables:
```sh
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Add your HF_TOKEN in .env.example and rename it to .env

Run the vllm server with:

```sh
nohup uv run vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
    --max-model-len 8192 \
    --enable-lora \
    --max-lora-rank 128 \
    --port 8000 > logs/vllm.log 2>&1 &
```

After Starting Your vLLM server, Run the proxy server with:
```sh
nohup uv run python -m autoload.server > proxy.log 2>&1 &
```


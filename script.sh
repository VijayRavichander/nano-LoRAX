#!/bin/bash

#Start Server
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-model-len 8192 \
    --enable-lora \
    --max-lora-rank 256 \
    --port 8000

# Hitting the Base Model
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "HuggingFaceTB/SmolLM2-135M-Instruct",
"messages": [
{
"role": "user",
"content": "How many players are on a touch rugby team?"
}
],
"max_tokens": 50,
"temperature": 0.01
}' | jq

# Hitting the Adapter 
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "jekunz/smollm-135m-lora-fineweb-swedish",
"messages": [
{
"role": "user",
"content": "How many players are on a touch rugby team?"
}
],
"max_tokens": 50,
"temperature": 0.01
}' | jq


# Meta Data Reqs
curl http://localhost:8080/v1/models | jq
curl http://localhost:8080/v1/lora/loaded | jq
curl http://localhost:8080/v1/lora/downloaded | jq
curl http://localhost:8080/v1/lora/status | jq



#!/bin/bash

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
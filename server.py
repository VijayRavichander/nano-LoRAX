# Proxy Serve used to handle LoRax Downloading and Loading into the vllm SERVER

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import uvicorn
from config import settings
import httpx

app = FastAPI()


vllm_client = httpx.AsyncClient(base_url = settings.VLLM_SERVER_URL)

@app.on_event("startup")
async def startup_event():
    '''Init Redis Manager'''
    try:
        print("Init Redis State")
    
    except Exception as e:
        print(f"Warning: Failed to Initialize Redis: {str(e)}")
        print("Server will start but LoRA management will fail")


# TODO - Desing Choice of choosing the right naming scheme
async def is_lora_model(model_name: str) -> bool:
    return 'adapter' in model_name.lower() 

async def forward_req_to_vllm(request: Request):

    try:
        url = f"{settings.VLLM_SERVER_URL}{request.url.path}"
        headers = dict(request.headers)
        method = request.method

        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, headers=headers, timeout=30.0)
            else:
                # For POST/PUT etc, forward the raw body
                body = await request.body() 
                response = await client.request(
                    method,
                    url,
                    content=body,  # Use content instead of json
                    headers=headers,
                    timeout=30.0
                )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        
    except Exception as e:
        print(f"Error forwarding request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    try: 
        body = await request.json()
        return await forward_req_to_vllm(request)

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detial = str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PROXY_PORT) 


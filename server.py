# Proxy Serve used to handle LoRax Downloading and Loading into the vllm SERVER

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import uvicorn
from config import settings
import httpx
from lora_manager import LoraManager
from redis_manager import RedisManager
import asyncio

app = FastAPI()
lora_manager = LoraManager()
redis_manager = RedisManager()

vllm_client = httpx.AsyncClient(base_url = settings.VLLM_SERVER_URL)


@app.on_event("startup")
async def startup_event():
    '''Init Redis Manager'''
    try:
        redis_manager.clear_all_state()
    except Exception as e:
        print(f"Warning: Failed to Initialize Redis: {str(e)}")
        print("Server will start but LoRA management will fail")


# TODO - Desing Choice of choosing the right naming scheme
async def is_lora_model(model_name: str) -> bool:
    return 'adapter' in model_name.lower() or 'lora' in model_name.lower()

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

        model_name = body.get('model')

        if not model_name:
            return HTTPException(status_code=400, detail="Please Include a Model Name")

        if await is_lora_model(model_name):

            hf_token = request.headers.get("X-HF-Token")
            success = await lora_manager.ensure_lora_loaded(model_name, hf_token)


        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await forward_req_to_vllm(request)
            except HTTPException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 * (attempt + 1))

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail = str(e))


@app.get("/v1/models")
async def list_models(request: Request):
    return await forward_req_to_vllm(request)


@app.get("/v1/lora/loaded")
async def get_loaded_loras():
    return {"loaded_loras": redis_manager.get_loaded_loras()}

@app.get("/v1/lora/downloaded")
async def get_downloaded_loras():
    return {"downloaded_loras": redis_manager.get_downloaded_loras()}

@app.get("/v1/lora/status")
async def get_status():
    stats = redis_manager.get_stats()
    return {
        "stats": stats, 
        "limits": {
            "max_loaded": settings.MAX_LOADED_LORAS, 
            "max_downloaded": settings.MAX_DOWNLOADED_LORAS
        }
    }

@app.api_route("/{path:path}", methods = ["GET", "POST", "PUT", "DELETE"])
async def proxy_endpoint(request: Request, path: str):
    return await forward_req_to_vllm(request)

async def cleanup():
    """Cleanup resources before shutdown"""
    print("Cleaning up resources...")
    # Close httpx clients
    await vllm_client.aclose()
    await lora_manager.cleanup()
    print("Cleanup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Handle graceful shutdown"""
    await cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PROXY_PORT) 


import os
import time
import httpx
from typing import Optional, Dict
from redis_manager import RedisManager
from config import settings
from collections import defaultdict
from asyncio import Lock
import asyncio
import shutil
from huggingface_hub import HfApi,  snapshot_download

class LoraManager:

    def __init__(self):
        self.redis = RedisManager()
        self.vllm_client = httpx.AsyncClient(base_url = settings.VLLM_SERVER_URL,
                                             timeout = 30.0)
        
        self.lora_locks = defaultdict(Lock)

    async def ensure_lora_loaded(self, lora_name: str, hf_token: Optional[str] = None) -> bool:
        
        if self.redis.is_loaded_lora(lora_name):

            print(f"LoRA {lora_name} already loaded, updating timestamp")
            self.redis.update_lora_timestamp(lora_name)
            return True

        async with self.lora_locks[lora_name]:

            if self.redis.is_loaded_lora(lora_name):
                print(f"LoRA {lora_name} already loaded, updating timestamp")
                self.redis.update_lora_timestamp(lora_name)
                return True
            
            try:
                if not self.redis.is_lora_downloaded(lora_name):
                    print("Try to donwload the LoRA adapter")
                    success = await self._download_lora(lora_name, hf_token)
                    if not success:
                        return False
                
                success = await self._load_lora(lora_name)
                print("Loading LoRA")
                return success
                
            
            except Exception as e:
                return False
            
    async def _download_lora(self, lora_name: str, hf_token: Optional[str] = None) -> None:
        try:
            print(f"Downloading LoRA: {lora_name}...")
            while (len(self.redis.get_downloaded_loras()) >= settings.MAX_DOWNLOADED_LORAS):
                lru_lora = self.redis.get_lru_downloaded_lora()
                if lru_lora:
                    await self._delete_lora(lru_lora)
            
            token = hf_token or settings.DEFAULT_HF_TOKEN
            print(f"Token: {token}")

            try:
                api = HfApi(token = token)
                print("API: ", api)
                try:
                    api.repo_info(repo_id = lora_name)

                except Exception as e:
                    if "401" in str(e):
                        print(f"Unauthorized access to {lora_name}. Token required.")
                        return False
                    raise

                adapter_path = snapshot_download(
                    lora_name,
                    cache_dir=settings.HF_CACHE_DIR,
                    token=token,
                    ignore_patterns=['model-*', 'pytorch_model*', 'tf_model*', 'flax_model*']
                )

                self.redis.record_lora_donwloaded(lora_name)
                print(f"Successfully downloaded LoRA {lora_name}")
                return True
                
            except Exception as e:
                if "401" in str(e):
                    print(f"Unauthorized access to {lora_name}. Token required.")
                    return False
                raise
                
        except Exception as e:
            print(f"Error downloading LoRA {lora_name}: {str(e)}")
            return False

    async def _verfication_lora_loaded(self, lora_name: str):
        try: 

            test_request = {
                "model": lora_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1  # Minimal generation to verify
            }

            response = await self.vllm_client.post(
                "/v1/chat/completions",
                json=test_request
            )

            if response.status_code == 200:
                print(f"Test inference successful for {lora_name}")
                return True
            
            print(f"Test inference failed for {lora_name}. Status: {response.status_code}, Response: {response.text}")
            return False
        
        except Exception as e:
            print(f"Error during test inference for {lora_name}: {str(e)}")
            return False

    async def _delete_lora(self, lora_name: str) -> None:

        try:
            path = self._get_lora_path(lora_name)
            if os.path.exists(path):
                shutil.rmtree(path)
            
            self.redis.record_lora_deleted(lora_name)
        
        except Exception as e:
            print(f"Error deleting LoRA {lora_name}: {str(e)}")

    def _get_lora_path(self, lora_name: str) -> None:
        print("Getting LoRA Path")
        try: 
            cache_dir = settings.HF_CACHE_DIR or os.path.expanduser("~/.cache/huggingface/hub")

            if isinstance(lora_name, bytes):
                lora_name = lora_name.decode()
            
            formatted_name = lora_name.replace('/', '--')
            repo_dir = os.path.join(cache_dir, f"models--{formatted_name}")

            if os.path.exists(repo_dir):
                refs_path = os.path.join(repo_dir, "refs")
                if os.path.exists(refs_path):
                    with open(os.path.join(refs_path, "main")) as f:
                        revision = f.read().strip()
                    return os.path.join(repo_dir, "snapshots", revision)
                return repo_dir
        
        except Exception as e:
            print(f"Error Formatting LoRA Path for {lora_name}: {str(e)}")
            return str(lora_name)

    async def _load_lora(self, lora_name: str):
        try:

            while(self.redis.get_loaded_loras().__len__() > settings.MAX_LOADED_LORAS):

                lru_lora = self.redis.get_lru_loaded_lora()

                if lru_lora:
                    if isinstance(lru_lora, bytes):
                        lru_lora = lru_lora.decode()
                    
                    print(f"Unloading LRU LoRA {lru_lora}")
                    await self._unload_lora(lru_lora)
            
            lora_path = self._get_lora_path(lora_name)
            print(f"Loading LoRA {lora_name} from path: {lora_path}")

            # Load the LORA
            response = await self.vllm_client.post(
                "/v1/load_lora_adapter",
                json = {
                    "lora_name": lora_name,
                    "lora_path": lora_path
                }
            )

            print(f"vLLM Loading Response from the Server for {lora_name}: status: {response} data: {response.text}")
            
            
            if response.status_code == 200 or (
                response.status_code == 400 and 
                any(msg in response.text.lower() for msg in ["already been loaded", "already beenloaded"])
            ):

                is_already_loaded = "already" in response.text.lower()
                print(f"vLLM status for {lora_name} : {'Already Loaded' if is_already_loaded else 'Loaded Successfully'}")
                await asyncio.sleep(0.1)

                if is_already_loaded:
                    self.redis.record_lora_loaded(lora_name)
                    return True
            
                # Verfication of LoRA Loading
                verification_lora = await self._verfication_lora_loaded(lora_name)

                if verification_lora: 
                    self.redis.record_lora_loaded(lora_name)
                    return True
            
            print(f"Unexpected response loading LoRA {lora_name}. Status: {response.status_code}, Response: {response.text}")
            return False


        except Exception as e: 
            print(f"Error loading LoRA {lora_name}: {str(e)}")
            return False
        
    async def _unload_lora(self, lora_name: str):

        try:
            
            if isinstance(lora_name, bytes):
                lora_name = lora_name.decode()
            
            response = await self.vllm_client.post(
                'v1/unload_lora_adapter', 
                json = {"lora_name": lora_name}
            )

            if response.status_code == 200:
                self.redis.record_lora_unloaded(lora_name)
                return True
            
            print(f"Failed to unload LoRA: {lora_name}")
            return False

        except Exception as e:
            print(f"Error while loading LoRA {lora_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    async def cleanup(self):
        try:
            # Unload all loaded LoRAs
            loaded_loras = self.redis.get_loaded_loras()
            for lora in loaded_loras:
                await self._unload_lora(lora["name"])
            
            # Close httpx client
            await self.vllm_client.aclose()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
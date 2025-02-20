import os
import time
import httpx
from typing import Optional, Dict
from redis import RedisManager
from config import settings
from collections import defaultdict
from asyncio import Lock
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

            if self.redis.is_lora_loaded(lora_name):
                print(f"LoRA {lora_name} already loaded, updating timestamp")
                self.redis.update_lora_timestamp(lora_name)
                return True
            
            try:

                if not self.redis.is_lora_downloaded(lora_name):
                    success = await self._download_lora(lora_name, hf_token)
                    if not success:
                        return False
                
            
            except Exception as e:
                return False
            
    async def download_lora(self, lora_name: str, hf_token: Optional[str] = None) -> None:
        try:
            print(f"Downloading LoRA: {lora_name}...")

            while (self.redist.get_downloaded_loras >= settings.MAX_DOWNLOADED_LORAS):
                lru_lora = self.redis.get_lru_downloaded_lora()
                if lru_lora:
                    await self._delete_lora(lru_lora)
            
            token = hf_token or settings.DEFAULT_HF_TOKEN

            try:
                api = HfApi(token = token)

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

                self.redis.record_lora_downloaded(lora_name)
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


    async def _delete_lora(self, lora_name: str) -> None:

        try:
            path = self.get_lora_path(lora_name)
            if os.path.exists(path):
                shutil.rmtree(path)
            
            self.redis.record_lora_deleted(lora_name)
        
        except Exception as e:
            print(f"Error deleting LoRA {lora_name}: {str(e)}")

    async def get_lora_path(self, lora_name: str) -> None:

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

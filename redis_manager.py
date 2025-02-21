import redis
import time
from config import settings
from typing import List, Dict, Optional

class RedisManager:

    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL)
        self.LOADED_LORAS = "loaded_loras" # Hash: LoRa_Name -> TimeStamp
        self.DOWNLOADED_LORAS = "downloaded_loras" # Hash: LoRa_Name -> TimeStamp
        self.STATS = "lora_stats"

    def init_stats(self):
        print("Init Stats in Redis")
        default_stats = {
            "loaded_count": 0, 
            "unloaded_count": 0, 
            "donwloaded_count": 0, 
            "deleted_count": 0
        }

        for key, value in default_stats.items():
            self.redis.hsetnx(self.STATS, key, value)
    

    def update_lora_timestamp(self,lora_name: str, is_loaded: bool = True):
        key = self.LOADED_LORAS if is_loaded else self.DOWNLOADED_LORAS
        self.redis.hset(key, lora_name, str(time.time()))

    
    def get_loaded_loras(self) -> List[Dict[str, float]]:
        loaded = self.redis.hgetall(self.LOADED_LORAS)
        
        loaded_loras = [{"name": name.decode(), "last_used": float(timestamp.decode())}
                         for name, timestamp in loaded.items()]

        return loaded_loras


    def get_downloaded_loras(self) -> List[Dict[str, float]]:
        donwloaded = self.redis.hgetall(self.DOWNLOADED_LORAS)
        
        donwloaded_loras = [{"name": name.decode(), "last_used": float(timestamp.decode())}
                         for name, timestamp in donwloaded.items()]

        return donwloaded_loras

    def is_loaded_lora(self, lora_name) -> bool:
        print(f"Checking if LORA Exist: {lora_name}")
        return self.redis.hexists(self.LOADED_LORAS, lora_name)

    def is_lora_downloaded(self, lora_name) -> bool:
        return self.redis.hexists(self.DOWNLOADED_LORAS, lora_name)
    
    def get_lru_loaded_lora(self) -> Optional[str]:
        loaded = self.get_loaded_loras()

        if not loaded: 
            return None
        
        lru = min(loaded, key = lambda x: x["last_used"])["name"]

        if isinstance(lru, bytes):
            lru = lru.decode()
        
        return lru

    def get_lru_downloaded_lora(self) -> Optional[str]:

        downloaded = self.get_downloaded_loras()

        if not downloaded:
            return None

        lru = min(downloaded, key = lambda x: x['last_used'])['name']

        if isinstance(lru, bytes):
            lru = lru.decode()
        
        return lru

    def record_lora_loaded(self, lora_name: str) -> None:
        self.update_lora_timestamp(lora_name, is_loaded=True)
        self.redis.hincrby(self.STATS, "loaded_count", 1)
    
    def record_lora_unloaded(self, lora_name: str) -> None:
        self.redis.hdel(self.LOADED_LORAS, lora_name)
        self.redis.hincrby(self.STATS, 'unloaded_count', 1)

    def record_lora_donwloaded(self, lora_name: str) -> None:
        self.update_lora_timestamp(lora_name, is_loaded=False)
        self.redis.hincrby(self.STATS, "donwloaded_count", 1)
    
    def record_lora_deleted(self, lora_name: str) -> None:
        self.redis.hdel(self.DOWNLOADED_LORAS, lora_name)
        self.redis.hincrby(self.STATS, 'deleted_count', 1)
    
    def get_stats(self) -> Dict:
        stats = self.redis.hgetall(self.STATS)
        return {k: int(v) for k, v in stats.items()}

    def clear_all_state(self):
        self.redis.delete(self.LOADED_LORAS)
        self.redis.delete(self.DOWNLOADED_LORAS)
        self.init_stats()



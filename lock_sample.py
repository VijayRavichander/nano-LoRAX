import asyncio
from collections import defaultdict

class LoRALockManager:
    def __init__(self):
        self.lora_locks = defaultdict(asyncio.Lock)  # Creates a lock per unique name

    async def process_lora(self, name):
        async with self.lora_locks["x"]:  # Only one coroutine per 'name' can enter at a time
            print(f"🔒 {name} - Lock acquired")
            await asyncio.sleep(2)  # Simulate a long-running task
            print(f"✅ {name} - Lock released")

async def main():
    manager = LoRALockManager()

    # Start two tasks with the same name
    await asyncio.gather(
        manager.process_lora("task1"),
        manager.process_lora("task4"),  # Will wait for the first one to finish
        manager.process_lora("task2")   # Can run in parallel since it has a different name
    )

asyncio.run(main())

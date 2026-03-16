import asyncio
from src.watcher import S3Watcher
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    watcher = S3Watcher()
    try:
        asyncio.run(watcher.start())
    except KeyboardInterrupt:
        print("Watcher stopped by user.")

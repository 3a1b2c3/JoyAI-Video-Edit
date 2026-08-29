#!/usr/bin/env python3
import asyncio
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        for i in range(10):
            await asyncio.sleep(1)
            await websocket.send_text(f"ping_{i}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

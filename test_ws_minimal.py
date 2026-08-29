#!/usr/bin/env python3
"""Minimal WebSocket test server - no model loading."""
import asyncio
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="JoyAI Test Server")

# Serve static files
import os
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "deploy")
if os.path.exists(os.path.join(DEPLOY_DIR, "static")):
    app.mount("/", StaticFiles(directory=os.path.join(DEPLOY_DIR, "static"), html=True), name="static")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "not_loaded"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("[WS] Connection attempt...")
    await websocket.accept()
    print("[WS] Connection accepted")

    try:
        count = 0
        while True:
            # Send heartbeat every 1 second
            await asyncio.sleep(1)
            count += 1
            msg = f"heartbeat_{count}"
            print(f"[WS] Sending: {msg}")
            await websocket.send_text(msg)
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")

if __name__ == "__main__":
    print("[SERVER] Starting minimal test server on 0.0.0.0:8080...")
    print("[SERVER] WebSocket endpoint: ws://localhost:8080/ws")
    print("[SERVER] Health endpoint: http://localhost:8080/health")

    uvicorn.run(app, host="0.0.0.0", port=8080)

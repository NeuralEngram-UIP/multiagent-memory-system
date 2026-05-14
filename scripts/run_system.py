"""
Quick-start script to spin up the multi-agent system locally.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

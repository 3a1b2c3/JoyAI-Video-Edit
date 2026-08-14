#!/usr/bin/env python3
"""Direct server launcher - bypasses module import issues."""

import sys
from pathlib import Path

# Add deploy to path
deploy_dir = Path(__file__).parent / "deploy"
sys.path.insert(0, str(deploy_dir))

# Now import and run
from xvideo.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

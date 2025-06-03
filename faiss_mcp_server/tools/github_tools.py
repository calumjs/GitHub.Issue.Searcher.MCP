import asyncio
import os
import logging
from typing import Dict, Any

from server import mcp
from utils.faiss_handler import faiss_handler # Singleton instance
from utils.github_syncer import perform_github_issue_sync

logger = logging.getLogger(__name__)

# Load GitHub PAT from environment
GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# GitHub sync functionality is available via command line only:
# python main.py --sync-repo owner/repo
# 
# This is intentionally not exposed as an MCP tool because:
# 1. It takes a very long time to complete
# 2. It blocks the MCP server during execution
# 3. It's meant for initial setup, not regular use

# --- Removed the original sync logic as it's now in github_syncer.py --- 
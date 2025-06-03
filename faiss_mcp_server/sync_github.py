#!/usr/bin/env python3
"""
Standalone GitHub Repository Sync Script for FAISS MCP Server

This script allows you to sync GitHub repository issues without starting the MCP server.
Useful for initial setup or periodic updates.

Usage:
    python sync_github.py owner/repo
    python sync_github.py facebookresearch/faiss
    python sync_github.py microsoft/vscode
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from utils.faiss_handler import faiss_handler
from utils.github_syncer import perform_github_issue_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sync GitHub repository issues to FAISS index")
    parser.add_argument("repo", help="Repository in format 'owner/repo'")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before syncing")
    
    args = parser.parse_args()
    
    # Validate repository format
    if '/' not in args.repo:
        logger.error(f"Invalid repository format: '{args.repo}'. Expected 'owner/repo'.")
        sys.exit(1)
    
    repo_owner, repo_name = args.repo.split('/', 1)
    
    # Get GitHub token
    github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_pat:
        logger.error("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")
        sys.exit(1)
    
    try:
        # Perform the sync
        result = perform_github_issue_sync(
            repo_owner=repo_owner,
            repo_name=repo_name,
            faiss_handler_instance=faiss_handler,
            github_pat=github_pat,
            clear_first=args.clear
        )
        
        logger.info(f"✅ {result}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
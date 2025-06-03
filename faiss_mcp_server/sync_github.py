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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('github_sync.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Sync GitHub repository issues to FAISS index",
        epilog="Example: python sync_github.py facebookresearch/faiss"
    )
    parser.add_argument(
        'repository',
        help='GitHub repository in format owner/repo (e.g., facebookresearch/faiss)'
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear the existing index before syncing (default: append to existing data)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from: {env_path}")
    else:
        logger.warning(f"Environment file not found: {env_path}")
        load_dotenv()  # Try default locations
    
    # Validate repository format
    if '/' not in args.repository:
        logger.error(f"Invalid repository format: '{args.repository}'. Expected 'owner/repo'.")
        sys.exit(1)
    
    repo_owner, repo_name = args.repository.split('/', 1)
    
    # Check for required environment variables
    github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not github_pat:
        logger.error("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        sys.exit(1)
    
    if not openai_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        sys.exit(1)
    
    logger.info("🚀 Starting GitHub Repository Sync")
    logger.info(f"Repository: {args.repository}")
    logger.info(f"Owner: {repo_owner}")
    logger.info(f"Name: {repo_name}")
    
    try:
        # Initialize FAISS handler
        logger.info("Initializing FAISS handler...")
        # The handler is automatically initialized as a singleton
        
        # Perform the sync
        result = perform_github_issue_sync(
            repo_owner=repo_owner,
            repo_name=repo_name,
            faiss_handler_instance=faiss_handler,
            github_pat=github_pat,
            clear_first=args.clear
        )
        
        logger.info("🎉 Sync completed!")
        logger.info(f"Result: {result}")
        print(f"\n✅ {result}")
        
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
        print("\n⚠️ Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        print(f"\n❌ Sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
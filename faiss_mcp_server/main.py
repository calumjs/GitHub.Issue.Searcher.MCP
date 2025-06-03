import logging
import argparse # Import argparse
import os       # Import os
import sys      # Import sys
import signal   # For graceful shutdown
from dotenv import load_dotenv # Import load_dotenv

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr instead of stdout for MCP compatibility
        logging.FileHandler('faiss_mcp_server.log', mode='a')
    ]
)

from server import mcp # mcp is an instance of FastMCP

# Import tools modules to register the tools decorated with @mcp.tool()
import tools.faiss_tools # noqa: F401 - Tools registered via import
import tools.github_tools # noqa: F401 - Added GitHub tools import

# Import the synchronous sync function
from utils.github_syncer import perform_github_issue_sync

logger = logging.getLogger(__name__)

# Load .env early for PAT check if needed
load_dotenv()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    try:
        from utils.faiss_handler import faiss_handler
        logger.info("Saving FAISS index before shutdown...")
        faiss_handler.save()
        logger.info("Shutdown complete.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # --- Add Argument Parsing --- 
    parser = argparse.ArgumentParser(description="FAISS MCP Server with optional GitHub sync.")
    parser.add_argument(
        '--sync-repo',
        type=str,
        metavar='OWNER/REPO',
        help='Specify a GitHub repository (owner/repo) to sync issues from on startup.'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear the existing index before syncing (only used with --sync-repo).'
    )
    args = parser.parse_args()
    # --------------------------

    logger.info(f"Starting MCP server '{mcp.name}'...")
    logger.info("Tools should be registered via imports.")

    # Initialize the FaissHandler singleton by accessing it
    try:
        # This import also triggers FaissHandler initialization
        from utils.faiss_handler import faiss_handler
        logger.info("FAISS Handler initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS Handler: {e}", exc_info=True)
        # If FAISS handler fails, we likely can't sync or run server properly
        sys.exit(1) # Exit if handler fails

    # --- Perform Sync if Argument Provided ---
    if args.sync_repo:
        repo_arg = args.sync_repo
        if '/' not in repo_arg:
            logger.error(f"Invalid format for --sync-repo: '{repo_arg}'. Expected 'owner/repo'.")
            sys.exit(1)
        
        repo_owner, repo_name = repo_arg.split('/', 1)
        logger.info(f"--sync-repo argument provided. Attempting sync for {repo_owner}/{repo_name}...")
        
        # Get PAT from environment (ensure .env is loaded)
        github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not github_pat:
            logger.error("Cannot perform sync: GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")
            sys.exit(1)

        # Call the synchronous function directly
        sync_status = perform_github_issue_sync(
            repo_owner=repo_owner,
            repo_name=repo_name,
            faiss_handler_instance=faiss_handler, # Use the initialized singleton
            github_pat=github_pat,
            clear_first=args.clear
        )
        # Log status instead of printing to avoid MCP protocol interference
        logger.info(f"GitHub Sync Status: {sync_status}")
        # Optionally exit if sync failed?
        # if "Error:" in sync_status:
        #     sys.exit(1)
    # ----------------------------------------

    # Run the MCP server (this will block until stopped)
    logger.info("Starting MCP server run loop...")
    mcp.run() 
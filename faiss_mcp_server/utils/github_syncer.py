import logging
import os
from github import Github, GithubException, RateLimitExceededException, UnknownObjectException, BadCredentialsException
from typing import Dict, Any

# Assuming faiss_handler is passed as an argument, so no direct import here
# from .faiss_handler import FaissHandler # This would be a circular dependency if FaissHandler also imported something from here

logger = logging.getLogger(__name__)

def perform_github_issue_sync(
    repo_owner: str,
    repo_name: str,
    faiss_handler_instance, # Pass the already initialized FaissHandler instance
    github_pat: str,
    clear_first: bool = True
) -> str:
    """
    Synchronously fetches issues from a GitHub repository, optionally clears the FAISS index,
    embeds issues, and saves them.

    Args:
        repo_owner: The owner of the GitHub repository.
        repo_name: The name of the GitHub repository.
        faiss_handler_instance: An initialized instance of FaissHandler.
        github_pat: GitHub Personal Access Token.
        clear_first: Whether to clear the existing index before syncing (default: True).

    Returns:
        A summary message indicating the outcome of the sync.
    """
    if not github_pat:
        return "Error: GitHub Personal Access Token not provided for sync."

    full_repo_name = f"{repo_owner}/{repo_name}"
    logger.info(f"[Sync Function] Starting GitHub issue sync for repository: {full_repo_name}")

    # 1. Optionally clear existing index
    if clear_first:
        logger.info("[Sync Function] Attempting to clear existing FAISS index before sync...")
        clear_success = faiss_handler_instance.clear_index()
        if not clear_success:
            logger.error("[Sync Function] Failed to clear the FAISS index. Aborting sync.")
            return "Error: Failed to clear the existing FAISS index before starting sync."
        logger.info("[Sync Function] Existing FAISS index cleared.")
    else:
        logger.info("[Sync Function] Skipping index clear - appending to existing data.")

    added_count = 0
    failed_count = 0
    issues_processed = 0

    try:
        g = Github(github_pat)
        try:
            _ = g.get_user().login
            logger.info("[Sync Function] GitHub token authenticated.")
        except BadCredentialsException:
            logger.error("[Sync Function] GitHub PAT is invalid or expired.")
            return "Error: Invalid or expired GitHub Personal Access Token."

        repo = g.get_repo(full_repo_name)
        logger.info(f"[Sync Function] Accessing repo: {repo.full_name}")

        issues = repo.get_issues(state='all')
        logger.info("[Sync Function] Fetching issues...")

        for issue in issues:
            issues_processed += 1
            if issues_processed % 50 == 0:
                logger.info(f"[Sync Function] Processed {issues_processed} issues...")

            try:
                # Create focused text for embedding - prioritize title and beginning of body
                title_text = f"Issue #{issue.number}: {issue.title}"
                body_text = issue.body or ''
                
                # For very long issues, prioritize the most important content
                if len(body_text) > 10000:  # If body is very long
                    # Take first 8000 chars of body to preserve the most important context
                    body_text = body_text[:8000] + "\n\n[Issue content continues...]"
                
                text_to_embed = f"{title_text}\n\n{body_text}"
                
                metadata = {
                    "source": "github_issue",
                    "repo": full_repo_name,
                    "issue_number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "creator": issue.user.login if issue.user else None,
                    "url": issue.html_url,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                    "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                    "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
                    "labels": [label.name for label in issue.labels],
                    "comments_count": issue.comments,
                    "original_body_length": len(issue.body) if issue.body else 0
                }

                success, msg = faiss_handler_instance.add_embedding(text=text_to_embed, metadata=metadata)

                if success:
                    added_count += 1
                    logger.debug(f"[Sync Function] Added issue #{issue.number} to memory.")
                else:
                    failed_count += 1
                    logger.warning(f"[Sync Function] Failed to add issue #{issue.number}: {msg}")

            except RateLimitExceededException:
                logger.error("[Sync Function] GitHub API rate limit exceeded during issue processing. Aborting sync.")
                # Save whatever was managed so far
                faiss_handler_instance.save()
                return f"Error: GitHub rate limit exceeded after processing {issues_processed} issues. Synced {added_count}, failed {failed_count}."
            except Exception as e_inner:
                failed_count += 1
                logger.error(f"[Sync Function] Error processing issue #{issue.number}: {e_inner}", exc_info=True)

        logger.info(f"[Sync Function] Finished processing {issues_processed} issues.")

        logger.info("[Sync Function] Saving synced issues to FAISS index...")
        save_success = faiss_handler_instance.save()
        save_msg = "successfully" if save_success else "failed"
        logger.info(f"[Sync Function] Index save operation completed {save_msg}.")

        return f"Sync complete for {full_repo_name}. Processed: {issues_processed}, Added: {added_count}, Failed: {failed_count}. Save status: {save_msg}."

    except UnknownObjectException:
        logger.error(f"[Sync Function] Repository not found: {full_repo_name}")
        return f"Error: Repository '{full_repo_name}' not found or not accessible with the provided token."
    except RateLimitExceededException:
        logger.error("[Sync Function] GitHub API rate limit exceeded during initial connection or repo fetch.")
        return "Error: GitHub rate limit exceeded."
    except GithubException as e_gh:
        logger.error(f"[Sync Function] GitHub API error during sync: {e_gh}", exc_info=True)
        return f"Error: A GitHub API error occurred: {e_gh.status} {e_gh.data}"
    except Exception as e_outer:
        logger.error(f"[Sync Function] Unexpected error during GitHub issue sync: {e_outer}", exc_info=True)
        faiss_handler_instance.save() # Attempt save
        return f"Error: An unexpected error occurred during sync. Details: {e_outer}" 
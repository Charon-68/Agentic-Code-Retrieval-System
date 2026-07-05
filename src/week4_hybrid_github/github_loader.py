"""GitHub repository ingestion with local fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader

logger = logging.getLogger(__name__)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_FALLBACK = PROJECT_ROOT / "sample_repo"


@dataclass(frozen=True)
class GitHubRepoConfig:
    """Configuration for GitHub repository ingestion."""

    owner: str
    repo: str
    branch: str = "main"
    github_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["GitHubRepoConfig"]:
        """Load GitHub repo settings from environment variables."""
        owner = os.getenv("GITHUB_REPO_OWNER", "").strip()
        repo = os.getenv("GITHUB_REPO_NAME", "").strip()
        token = os.getenv("GITHUB_TOKEN", "").strip() or None
        branch = os.getenv("GITHUB_BRANCH", "main").strip()

        if not owner or not repo:
            return None

        return cls(owner=owner, repo=repo, branch=branch, github_token=token)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.github_token)


def load_documents_from_github(config: GitHubRepoConfig) -> list[Document]:
    """
    Ingest a GitHub repository using LlamaIndex GithubRepositoryReader.

    Raises if the token is missing or the API call fails.
    """
    from llama_index.readers.github import GithubRepositoryReader, GithubRepositoryReaderRepoCfg

    if not config.github_token:
        raise ValueError("GITHUB_TOKEN is required for remote GitHub ingestion")

    logger.info(
        "Loading GitHub repo %s/%s (branch=%s)",
        config.owner,
        config.repo,
        config.branch,
    )

    github_client = GithubRepositoryReader(
        github_token=config.github_token,
        verbose=True,
    )

    repo_cfg = GithubRepositoryReaderRepoCfg(
        repo=config.repo,
        owner=config.owner,
        filter_file_extensions=(
            [".py", ".md", ".txt", ".yaml", ".yml", ".json"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_directories=(
            ["tests", "test", ".github", "node_modules", "__pycache__"],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
    )

    documents = github_client.load_data(repo_cfg)
    logger.info("Loaded %d documents from GitHub", len(documents))
    return documents


def load_documents_from_local(
    directory: Path = DEFAULT_LOCAL_FALLBACK,
    extensions: Optional[list[str]] = None,
) -> list[Document]:
    """Load source files from a local directory as LlamaIndex Documents."""
    if not directory.exists():
        raise FileNotFoundError(f"Local fallback directory not found: {directory}")

    ext = extensions or [".py", ".md", ".txt"]
    reader = SimpleDirectoryReader(
        input_dir=str(directory),
        required_exts=ext,
        recursive=True,
    )
    documents = reader.load_data()
    logger.info("Loaded %d documents from local path %s", len(documents), directory)
    return documents


def load_repository_documents(
    config: Optional[GitHubRepoConfig] = None,
    local_fallback: Path = DEFAULT_LOCAL_FALLBACK,
) -> tuple[list[Document], str]:
    """
    Load repository documents from GitHub or fall back to local sample_repo.

    Returns (documents, source_description).
    """
    cfg = config or GitHubRepoConfig.from_env()

    if cfg and cfg.is_authenticated:
        try:
            documents = load_documents_from_github(cfg)
            source = f"github:{cfg.owner}/{cfg.repo}@{cfg.branch}"
            return documents, source
        except Exception as exc:
            logger.warning(
                "GitHub ingestion failed (%s). Falling back to local: %s",
                exc,
                local_fallback,
            )
    else:
        logger.info(
            "No valid GitHub token or repo config. Using local fallback: %s",
            local_fallback,
        )

    documents = load_documents_from_local(local_fallback)
    return documents, f"local:{local_fallback}"

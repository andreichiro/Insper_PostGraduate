import asyncio
import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.parse import urlparse

from firecrawl import Firecrawl

# Base directory = directory where this script lives
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class CrawlerSettings:
    """
    Configuration for Firecrawl-based crawling.
    """
    api_key: str = field(default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY", ""))
    # By default, write output next to this script
    output_dir: Path = field(default_factory=lambda: BASE_DIR)
    # None => let Firecrawl use its own default (currently 10k pages per crawl)
    limit_per_root: Optional[int] = None
    scrape_formats: List[str] = field(default_factory=lambda: ["markdown"])
    only_main_content: bool = True
    poll_interval: int = 15  # seconds between status polls on Firecrawl side
    timeout_seconds: int = 1800  # overall timeout per crawl (waiter) in seconds

    # Concurrency / rate limiting
    concurrency: int = 2  # max concurrent crawl jobs
    max_retries: int = 5
    initial_backoff_seconds: float = 5.0
    backoff_multiplier: float = 2.0
    random_jitter_seconds: float = 1.0
    delay_between_jobs_seconds: float = 0.0  # optional fixed delay after each successful crawl

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY is not set. "
                "Set it in the environment or pass api_key to CrawlerSettings."
            )
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.limit_per_root is not None and self.limit_per_root < 1:
            raise ValueError("limit_per_root must be >= 1 when set")


def slugify(value: str, max_length: int = 80) -> str:
    """
    Simple filesystem-safe slugify.
    """
    value = value.strip().lower()
    # Replace non-alphanumeric characters with dashes
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "page"
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value


class FirecrawlCrawler:
    """
    High-level crawler around the Firecrawl Python SDK.

    - Uses the blocking `Firecrawl.crawl` waiter under the hood.
    - Orchestrated with asyncio to support multiple roots concurrently.
    - Handles retries with exponential backoff for transient errors.
    - Saves each page to disk as Markdown (+optional HTML) plus a JSON metadata index.
    """

    # Patterns of noisy UI/LLM-helper strings to strip from markdown
    UNWANTED_LINE_PATTERNS = [
        re.compile(r"skip to main content", re.IGNORECASE),
        re.compile(r"^copy page$", re.IGNORECASE),
        re.compile(r"copy page as markdown for llms", re.IGNORECASE),
        re.compile(r"open in chatgpt", re.IGNORECASE),
        re.compile(r"open in claude", re.IGNORECASE),
        re.compile(r"open in perplexity", re.IGNORECASE),
        re.compile(r"ask questions about this page", re.IGNORECASE),
    ]

    def __init__(self, settings: CrawlerSettings) -> None:
        settings.validate()
        self.settings = settings
        self.client = Firecrawl(api_key=settings.api_key)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _clean_markdown(self, markdown: str) -> str:
        """
        Heuristic cleaner to strip Firecrawl/docs UI noise from markdown
        while preserving core page content.
        """
        lines = markdown.splitlines()
        cleaned: List[str] = []

        skip_toc = False  # after "On this page", skip bullet TOC items

        for line in lines:
            stripped = line.strip()

            # Drop known noisy UI lines
            if any(p.search(stripped) for p in self.UNWANTED_LINE_PATTERNS):
                continue

            # Drop "On this page" TOC heading and subsequent bullet list
            if stripped.lower() == "on this page":
                skip_toc = True
                continue

            if skip_toc:
                # Skip bullet-style TOC entries
                if stripped.startswith("- ") or stripped.startswith("* "):
                    continue
                # First non-bullet line ends TOC; process that line normally
                skip_toc = False

            cleaned.append(line)

        return "\n".join(cleaned)

    async def crawl_many(self, urls: Iterable[str]) -> None:
        """
        Crawl multiple root URLs concurrently (with a concurrency limit).
        """
        url_list = [u.strip() for u in urls if u and u.strip()]
        if not url_list:
            self.logger.warning("No URLs provided to crawl.")
            return

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def worker(url: str) -> None:
            async with semaphore:
                await self._crawl_single_with_retries(url)

        tasks = [asyncio.create_task(worker(url)) for url in url_list]
        await asyncio.gather(*tasks)

    async def _crawl_single_with_retries(self, url: str) -> None:
        """
        Run a single `crawl` with retry + exponential backoff on transient errors.
        """
        attempt = 0
        delay = self.settings.initial_backoff_seconds

        while True:
            attempt += 1
            try:
                self.logger.info("Starting crawl for %s (attempt %d)", url, attempt)

                # Build kwargs for Firecrawl.crawl
                crawl_kwargs: dict[str, Any] = {
                    "url": url,
                    "scrape_options": {
                        "formats": self.settings.scrape_formats,
                        "onlyMainContent": self.settings.only_main_content,
                    },
                    "poll_interval": self.settings.poll_interval,
                    "timeout": self.settings.timeout_seconds,
                }
                if self.settings.limit_per_root is not None:
                    crawl_kwargs["limit"] = self.settings.limit_per_root

                # Use Firecrawl's waiter-style crawl (auto-pagination, blocking) in a thread.
                crawl_status = await asyncio.to_thread(self.client.crawl, **crawl_kwargs)

                status = getattr(crawl_status, "status", None)
                docs = getattr(crawl_status, "data", []) or []

                if status != "completed":
                    self.logger.warning(
                        "Crawl for %s finished with status '%s' (docs=%d)",
                        url,
                        status,
                        len(docs),
                    )
                else:
                    self.logger.info(
                        "Crawl for %s completed successfully (docs=%d)",
                        url,
                        len(docs),
                    )

                self._save_crawl_result(url, crawl_status)

                # Optional fixed delay after a successful crawl to be extra nice with rate limits
                if self.settings.delay_between_jobs_seconds > 0:
                    await asyncio.sleep(self.settings.delay_between_jobs_seconds)

                return

            except Exception as exc:  # Firecrawl raises its own exception types; catch-all here
                message = str(exc).lower()
                is_rate_limit = "429" in message or "rate limit" in message
                is_server_error = any(code in message for code in ("500", "502", "503", "504"))
                transient = is_rate_limit or is_server_error

                if attempt < self.settings.max_retries and transient:
                    backoff = delay + random.uniform(0, self.settings.random_jitter_seconds)
                    self.logger.warning(
                        "Transient error while crawling %s: %s. "
                        "Retrying in %.1f s (attempt %d/%d)...",
                        url,
                        exc,
                        backoff,
                        attempt,
                        self.settings.max_retries,
                    )
                    await asyncio.sleep(backoff)
                    delay *= self.settings.backoff_multiplier
                    continue

                # Non-transient error or retries exhausted
                self.logger.exception(
                    "Failed to crawl %s after %d attempt(s); giving up.",
                    url,
                    attempt,
                )
                return

    def _save_crawl_result(self, root_url: str, crawl_status: Any) -> None:
        """
        Persist crawl results for a root URL to disk.

        For each page:
        - <root_slug>/<index>_<slug>_<hash>.md  (markdown content)
        - <root_slug>/<index>_<slug>_<hash>.html (optional HTML if requested)
        - <root_slug>/<index>_<slug>_<hash>.json (metadata + pointers to files + raw document)
        Also writes:
        - <root_slug>/_index.json (manifest for the root)
        """
        docs = getattr(crawl_status, "data", []) or []
        if not docs:
            self.logger.warning("No documents returned for root %s", root_url)
            return

        parsed_root = urlparse(root_url)
        root_label = parsed_root.netloc or parsed_root.path or "root"
        root_slug = slugify(root_label)
        root_dir = self.settings.output_dir / root_slug
        root_dir.mkdir(parents=True, exist_ok=True)

        index_records: List[dict[str, Any]] = []

        for idx, doc in enumerate(docs, start=1):
            # Firecrawl Document has attributes like `.markdown`, `.html`, `.metadata`
            markdown = getattr(doc, "markdown", None)
            html = getattr(doc, "html", None)

            # Normalize metadata (DocumentMetadata -> dict)
            metadata_obj = getattr(doc, "metadata", None)
            if metadata_obj is None:
                metadata: dict[str, Any] = {}
            elif isinstance(metadata_obj, dict):
                metadata = metadata_obj
            elif hasattr(metadata_obj, "model_dump"):
                metadata = metadata_obj.model_dump()
            elif hasattr(metadata_obj, "dict"):
                metadata = metadata_obj.dict()
            else:
                try:
                    metadata = dict(metadata_obj)  # type: ignore[arg-type]
                except Exception:
                    metadata = getattr(metadata_obj, "__dict__", {}) or {}

            source_url = (
                metadata.get("sourceURL")
                or metadata.get("url")
                or metadata.get("source_url")
                or root_url
            )
            title = metadata.get("title") or source_url or f"{root_url}#{idx}"

            slug = slugify(str(title))
            short_hash = hashlib.sha1(str(source_url).encode("utf-8")).hexdigest()[:8]
            base_name = f"{idx:05d}_{slug}_{short_hash}"

            md_path = root_dir / f"{base_name}.md"
            html_path = root_dir / f"{base_name}.html"
            meta_path = root_dir / f"{base_name}.json"

            # Markdown content (always create the file, even if empty, to keep index consistent)
            raw_md = markdown or ""
            md_text = self._clean_markdown(raw_md)
            md_path.write_text(md_text, encoding="utf-8")

            # Optional HTML if it was requested via formats
            if "html" in self.settings.scrape_formats and html:
                html_path.write_text(html, encoding="utf-8")
                html_file_name: Optional[str] = html_path.name
            else:
                html_file_name = None

            # Try to capture the raw Firecrawl Document as a dict for completeness
            if hasattr(doc, "model_dump"):
                raw_doc = doc.model_dump()  # type: ignore[call-arg]
            elif hasattr(doc, "dict"):
                raw_doc = doc.dict()  # type: ignore[call-arg]
            elif hasattr(doc, "__dict__"):
                raw_doc = vars(doc)
            else:
                raw_doc = {"value": str(doc)}

            record: dict[str, Any] = {
                "file_base": base_name,
                "markdown_file": md_path.name,
                "html_file": html_file_name,
                "source_url": source_url,
                "title": title,
                "metadata": metadata,
            }
            index_records.append(record)

            # Metadata + raw doc per page
            meta_payload = {
                "root_url": root_url,
                "source_url": source_url,
                "title": title,
                "markdown_file": md_path.name,
                "html_file": html_file_name,
                "metadata": metadata,
                "raw_document": raw_doc,
            }
            meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        index_payload = {
            "root_url": root_url,
            "total_documents": len(index_records),
            "documents": index_records,
        }
        index_path = root_dir / "_index.json"
        index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info(
            "Saved %d document(s) for root %s into %s",
            len(index_records),
            root_url,
            root_dir,
        )


async def run_crawls(urls: Iterable[str], settings: Optional[CrawlerSettings] = None) -> None:
    """
    Convenience entry point to crawl multiple URLs with a given settings object.
    """
    if settings is None:
        settings = CrawlerSettings()
    crawler = FirecrawlCrawler(settings)
    await crawler.crawl_many(urls)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Crawl multiple URLs with Firecrawl and save results to disk.",
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more root URLs to crawl (e.g. https://docs.firecrawl.dev).",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=BASE_DIR,
        help="Base output directory (default: directory of this script).",
    )
    parser.add_argument(
        "--concurrency",
        dest="concurrency",
        type=int,
        default=2,
        help="Maximum number of concurrent crawl jobs (default: 2).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["markdown"],
        help="Scrape formats to request from Firecrawl (e.g. markdown html). "
             "Defaults to: markdown",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    settings = CrawlerSettings(
        api_key=api_key or "",
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        scrape_formats=args.formats,
    )

    try:
        asyncio.run(run_crawls(args.urls, settings))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user; shutting down.")


# export FIRECRAWL_API_KEY="fc-XXXX"
# python firecrawl_crawler.py \
#   https://docs.firecrawl.dev https://www.firecrawl.dev \
#   --formats markdown html
# export FIRECRAWL_API_KEY="fc-c35b5687394b4c6ca4e536b18ef8929b"
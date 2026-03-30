"""HTML article/content extractor for mixed sources (Docusaurus, WordPress/Elementor, generic).

Requirements (install once):

    pip install beautifulsoup4 lxml markdownify readability-lxml

All third-party dependencies are optional except BeautifulSoup + lxml:
- If `markdownify` is missing, Markdown output will degrade to plain text.
- If `readability-lxml` is missing, generic extraction will fall back to a simple <body> heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
import re

from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as _markdownify
except ImportError:  # pragma: no cover - handled at runtime
    _markdownify = None

try:
    from readability import Document as _ReadabilityDocument
except ImportError:  # pragma: no cover
    _ReadabilityDocument = None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CleanResult:
    title: Optional[str]
    html: str           # cleaned inner HTML of main content (empty for markdown input)
    markdown: str       # markdown (if markdownify installed, else same as text or original markdown)
    text: str           # plain text-ish representation
    strategy: Literal["docusaurus", "elementor", "readability", "fallback", "markdown"]


@dataclass
class Section:
    heading: Optional[str]
    level: int
    paragraphs: List[str]


# ---------------------------------------------------------------------------
# Heuristics and configuration
# ---------------------------------------------------------------------------

GLOBAL_DROP_TAGS = {
    "script", "style", "noscript", "svg", "link", "meta",
    "iframe", "form", "button", "input", "textarea",
    "audio", "video", "canvas",
}

GLOBAL_DROP_SELECTORS = [
    "[role=banner]",          # global headers / announcement bars
    "[role=navigation]",      # sidebars, navs
    ".grecaptcha-badge",      # reCAPTCHA badge
    ".grecaptcha-error",
    "#forethought-chat",      # chat widget container
]

# Extra noise specific to dbt/docs Docusaurus pages
DOCUSAURUS_INTERNAL_DROP_SELECTORS = [
    ".copyPageContainer_v8EB",
    ".copyPageContainer_x6bT",
    ".feedbackContainer_bxwQ",
    ".tableOfContents_jeP5",
    ".announcementBar_s0pr",
    ".customSearchWeight_fp9e",
]

# Extra noise specific to the phData Elementor example
ELEMENTOR_INTERNAL_DROP_SELECTORS = [
    ".blog-cta-box-full",     # bottom CTA "Contact phData Today!"
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _remove_global_noise(soup: BeautifulSoup) -> None:
    # Drop obvious non-content tags everywhere
    for tag in soup.find_all(GLOBAL_DROP_TAGS):
        tag.decompose()

    # Drop well-known boilerplate containers
    for selector in GLOBAL_DROP_SELECTORS:
        for t in soup.select(selector):
            t.decompose()

    # Remove inline event handlers (onclick, onload, etc.)
    for attr in ("onload", "onclick", "onerror", "onmouseover", "onchange"):
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]


def _strip_attributes(node) -> None:
    """Strip most attributes to get clean structural HTML."""
    allowed_attrs = {"href", "src", "alt", "title"}
    for tag in node.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}


def _guess_title(node) -> Optional[str]:
    h1 = node.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    h2 = node.find("h2")
    if h2 and h2.get_text(strip=True):
        return h2.get_text(strip=True)

    return None


def _to_markdown(node) -> str:
    html = str(node)
    if _markdownify is None:
        # Fallback: plain text with basic line breaks
        return node.get_text(separator="\n", strip=True)
    return _markdownify(
        html,
        heading_style="ATX",
        strip=["span"],
    ).strip()


def _node_to_text(node) -> str:
    return node.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Site / layout detection
# ---------------------------------------------------------------------------

def _is_docusaurus(soup: BeautifulSoup) -> bool:
    # dbt/docs and many Docusaurus sites
    if soup.select_one(".theme-doc-markdown.markdown"):
        return True
    html_tag = soup.find("html")
    if html_tag and "docs-wrapper" in (html_tag.get("class") or []):
        return True
    return False


def _is_elementor_wordpress(soup: BeautifulSoup) -> bool:
    # phData-style Elementor single post
    if soup.select_one("main#article-content"):
        return True
    if soup.select_one('[data-elementor-type="wp-post"]'):
        return True
    body = soup.find("body")
    if body:
        classes = body.get("class") or []
        if "wp-theme-hello-elementor" in classes or "elementor-page" in classes:
            return True
    return False


# ---------------------------------------------------------------------------
# Site-aware extractors (HTML)
# ---------------------------------------------------------------------------

def _extract_docusaurus_main(soup: BeautifulSoup):
    # Prefer the markdown container inside the article
    main = soup.select_one("article .theme-doc-markdown.markdown")
    if not main:
        main = soup.select_one(".theme-doc-markdown.markdown")
    if not main:
        article = soup.find("article")
        if article:
            main = article
        else:
            main = soup.body or soup

    # Drop known non-content blocks inside docs area
    for selector in DOCUSAURUS_INTERNAL_DROP_SELECTORS:
        for t in main.select(selector):
            t.decompose()

    # Drop residual interactive bits
    for tag in main.select("button"):
        tag.decompose()

    _strip_attributes(main)
    return main


def _extract_elementor_main(soup: BeautifulSoup):
    # Blog post root
    main = soup.select_one("main#article-content")
    if not main:
        main = soup.select_one("[data-elementor-type='wp-post']")
    if not main:
        main = soup.body or soup

    # Narrow further to the wp-post container if present
    post = main.select_one('[data-elementor-type="wp-post"]') or main

    # Drop CTA and other known non-article blocks
    for selector in ELEMENTOR_INTERNAL_DROP_SELECTORS:
        for t in post.select(selector):
            t.decompose()

    # Drop code copy toolbars but keep <pre><code> content
    for toolbar in post.select(".toolbar"):
        toolbar.decompose()

    _strip_attributes(post)
    return post


def _extract_readability_main(html: str) -> Optional[BeautifulSoup]:
    if _ReadabilityDocument is None:
        return None
    try:
        doc = _ReadabilityDocument(html)
        main_html = doc.summary(html_partial=True)
    except Exception:
        return None
    soup = BeautifulSoup(main_html, "lxml")
    _remove_global_noise(soup)
    root = soup.body or soup
    _strip_attributes(root)
    return root


# ---------------------------------------------------------------------------
# Section-level structuring for HTML
# ---------------------------------------------------------------------------

def html_to_sections(clean_html: str) -> List[Section]:
    """Convert cleaned HTML into a list of logical sections.

    A section starts at each heading (h1-h6) and accumulates following
    paragraphs / list items / code blocks as its content.
    """
    soup = BeautifulSoup(clean_html, "lxml")
    sections: List[Section] = []
    current = Section(heading=None, level=0, paragraphs=[])

    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                             "p", "li", "pre", "code"]):
        name = el.name or ""
        if name.startswith("h"):
            # Close previous section
            if current.heading or current.paragraphs:
                sections.append(current)
            level = int(name[1]) if len(name) > 1 and name[1].isdigit() else 1
            current = Section(
                heading=el.get_text(" ", strip=True),
                level=level,
                paragraphs=[],
            )
        else:
            text = el.get_text(" ", strip=True)
            if text:
                current.paragraphs.append(text)

    if current.heading or current.paragraphs:
        sections.append(current)

    return sections


# ---------------------------------------------------------------------------
# Markdown-specific helpers
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _markdown_to_sections(markdown: str) -> List[Section]:
    """Very lightweight Markdown section splitter.

    - Headings (`#`, `##`, ...) start new sections.
    - All other lines are accumulated into paragraphs under the current section.
    """
    sections: List[Section] = []
    current = Section(heading=None, level=0, paragraphs=[])
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        paragraph = " ".join(line.strip() for line in buffer if line.strip())
        if paragraph:
            current.paragraphs.append(paragraph)
        buffer = []

    for line in markdown.splitlines():
        m = _MD_HEADING_RE.match(line.strip())
        if m:
            # Finish previous section
            flush_buffer()
            if current.heading or current.paragraphs:
                sections.append(current)
            level = len(m.group(1))
            heading = m.group(2).strip()
            current = Section(heading=heading, level=level, paragraphs=[])
        else:
            buffer.append(line)

    flush_buffer()
    if current.heading or current.paragraphs:
        sections.append(current)

    return sections


def _guess_title_from_markdown(markdown: str) -> Optional[str]:
    for line in markdown.splitlines():
        stripped = line.strip()
        m = _MD_HEADING_RE.match(stripped)
        if m:
            return m.group(2).strip()
    return None


def clean_markdown(markdown: str, source_hint: Optional[str] = None) -> CleanResult:
    """Clean a Markdown document using similar ideas as the HTML cleaner.

    Heuristics:
    - Drop "Skip to content" links.
    - Drop a leading hero image (first image-only line near the top).
    - Preserve the rest of the content as-is.
    """
    lines = markdown.splitlines()
    cleaned_lines: List[str] = []
    dropped_hero_image = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()

        # Skip accessibility / skip-link noise
        if lower.startswith("[skip to content]"):
            continue

        # Drop a single hero image near the very top
        if (
            not dropped_hero_image
            and idx < 10
            and stripped.startswith("![](")
        ):
            dropped_hero_image = True
            continue

        cleaned_lines.append(line)

    cleaned_md = "\n".join(cleaned_lines).strip()

    # Build sections for downstream chunking
    sections = _markdown_to_sections(cleaned_md)

    # Guess title from first heading
    title = _guess_title_from_markdown(cleaned_md)

    # Flatten sections into plain-text-ish representation
    text_blocks: List[str] = []
    for sec in sections:
        if sec.heading:
            text_blocks.append(sec.heading)
        text_blocks.extend(sec.paragraphs)
    text = "\n\n".join(text_blocks).strip()

    return CleanResult(
        title=title,
        html="",  # no HTML for markdown sources
        markdown=cleaned_md,
        text=text,
        strategy="markdown",
    )


# ---------------------------------------------------------------------------
# Public HTML API
# ---------------------------------------------------------------------------

def clean_html(
    html: str,
    source_hint: Optional[str] = None,
) -> CleanResult:
    """Clean a raw HTML document and return only the article-like content.

    Handles:
    - Docusaurus (dbt/docs-style) pages.
    - Elementor/WordPress posts (like the phData example).
    - Other sites via readability-lxml or a simple <body>-based fallback.
    """
    soup = BeautifulSoup(html, "lxml")
    _remove_global_noise(soup)

    # Strategy decision
    if _is_docusaurus(soup):
        strategy: Literal["docusaurus", "elementor", "readability", "fallback", "markdown"] = "docusaurus"
        node = _extract_docusaurus_main(soup)
    elif _is_elementor_wordpress(soup):
        strategy = "elementor"
        node = _extract_elementor_main(soup)
    else:
        readable = _extract_readability_main(html)
        if readable is not None:
            strategy = "readability"
            node = readable
        else:
            strategy = "fallback"
            node = soup.body or soup
        _strip_attributes(node)

    title = _guess_title(node)
    inner_html = "".join(str(c) for c in node.contents)
    markdown = _to_markdown(node)
    text = _node_to_text(node)

    return CleanResult(
        title=title,
        html=inner_html,
        markdown=markdown,
        text=text,
        strategy=strategy,
    )


def clean_html_file(path: Path) -> CleanResult:
    html = path.read_text(encoding="utf-8", errors="ignore")
    return clean_html(html, source_hint=str(path))


def process_html_file(path: Path, source_hint: Optional[str] = None) -> Dict[str, Any]:
    """ETL-style helper that returns a structured JSON-ready representation for HTML."""
    result = clean_html_file(path)
    sections = html_to_sections(result.html)
    return {
        "path": str(path),
        "source_hint": source_hint or str(path),
        "title": result.title,
        "strategy": result.strategy,
        "sections": [
            {
                "heading": s.heading,
                "level": s.level,
                "paragraphs": s.paragraphs,
            }
            for s in sections
        ],
        "markdown": result.markdown,
        "text": result.text,
    }


# ---------------------------------------------------------------------------
# Public Markdown API
# ---------------------------------------------------------------------------

def clean_markdown_file(path: Path) -> CleanResult:
    markdown = path.read_text(encoding="utf-8", errors="ignore")
    return clean_markdown(markdown, source_hint=str(path))


def process_markdown_file(path: Path, source_hint: Optional[str] = None) -> Dict[str, Any]:
    """ETL-style helper that returns a structured JSON-ready representation for Markdown."""
    result = clean_markdown_file(path)
    sections = _markdown_to_sections(result.markdown)
    return {
        "path": str(path),
        "source_hint": source_hint or str(path),
        "title": result.title,
        "strategy": result.strategy,
        "sections": [
            {
                "heading": s.heading,
                "level": s.level,
                "paragraphs": s.paragraphs,
            }
            for s in sections
        ],
        "markdown": result.markdown,
        "text": result.text,
    }


# ---------------------------------------------------------------------------
# Simple CLI (saves to disk instead of printing full content)
# ---------------------------------------------------------------------------

def _main_cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract main article-like content from HTML/Markdown files."
    )
    parser.add_argument("paths", nargs="+", help="HTML/Markdown file(s) to process")
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "text"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "markdown"],
        default="auto",
        help="Input type. 'auto' infers from file extension.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where cleaned files will be written. "
            "Default: alongside each input file."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in args.paths:
        path = Path(raw_path)

        if not path.exists():
            print(f"[WARN] Skipping missing file: {path}")
            continue

        suffix = path.suffix.lower()

        # Decide whether this is markdown or html
        if args.mode == "markdown":
            is_markdown = True
        elif args.mode == "html":
            is_markdown = False
        else:  # auto
            is_markdown = suffix in {".md", ".markdown"}

        # Compute content and extension
        if args.format == "json":
            if is_markdown:
                doc = process_markdown_file(path)
            else:
                doc = process_html_file(path)
            content = json.dumps(doc, ensure_ascii=False, indent=2)
            ext = ".json"
        else:
            if is_markdown:
                result = clean_markdown_file(path)
            else:
                result = clean_html_file(path)

            if args.format == "markdown":
                content = result.markdown
                ext = ".md"
            else:  # text
                content = result.text
                ext = ".txt"

        # Determine output path
        if output_dir is not None:
            output_path = output_dir / (path.stem + ext)
        else:
            output_path = path.with_suffix(ext)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        # Log a small status line (we never print the full text)
        print(f"[INFO] Wrote {args.format} for {path} -> {output_path}")


if __name__ == "__main__":
    _main_cli()

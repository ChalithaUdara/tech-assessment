"""
Utility script to download a 50MB+ public-domain literature corpus into `data/raw`.

The corpus mirrors the curated Project Gutenberg selection from the assessment brief.
Run with: `python scripts/download_corpus.py`
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests
from requests import Response
from tqdm import tqdm
from datacom_ai.utils.logger import logger

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_SLEEP_SECONDS = 0.5

# Author -> List[(book_id, title)]
BOOKS: Dict[str, Iterable[Tuple[int, str]]] = {
    "Charles_Dickens": [
        (98, "A Tale of Two Cities"),
        (1400, "Great Expectations"),
        (766, "David Copperfield"),
        (1023, "Bleak House"),
        (730, "Oliver Twist"),
        (580, "The Pickwick Papers"),
        (967, "Martin Chuzzlewit"),
        (968, "Nicholas Nickleby"),
        (821, "Dombey and Son"),
        (963, "Little Dorrit"),
        (883, "Our Mutual Friend"),
        (19337, "A Christmas Carol"),
    ],
    "Mark_Twain": [
        (76, "Huckleberry Finn"),
        (74, "Tom Sawyer"),
        (86, "A Connecticut Yankee in King Arthur's Court"),
        (119, "A Tramp Abroad"),
        (245, "Life on the Mississippi"),
        (3176, "The Innocents Abroad"),
        (3177, "Roughing It"),
        (1837, "The Prince and the Pauper"),
        (3178, "The Gilded Age"),
        (2572, "Pudd'nhead Wilson"),
    ],
    "Honore_de_Balzac": [
        (1968, "Pere Goriot"),
        (2172, "Cousin Bette"),
        (1967, "Eugenie Grandet"),
        (174, "The Magic Skin"),
        (13162, "Lost Illusions"),
        (1562, "Cousin Pons"),
        (1633, "Bureaucracy"),
        (1783, "Catherine de Medici"),
        (12915, "The Country Doctor"),
    ],
    "Leo_Tolstoy": [
        (2600, "War and Peace"),
        (1399, "Anna Karenina"),
        (262, "The Kreutzer Sonata"),
    ],
    "Victor_Hugo": [
        (135, "Les Miserables"),
        (3600, "The Hunchback of Notre Dame"),
        (1354, "Toilers of the Sea"),
        (9601, "The Man Who Laughs"),
    ],
    "Fyodor_Dostoevsky": [
        (28054, "The Brothers Karamazov"),
        (2554, "Crime and Punishment"),
        (2638, "The Idiot"),
        (8117, "The Possessed"),
        (600, "Notes from the Underground"),
    ],
    "George_Eliot": [
        (145, "Middlemarch"),
        (19690, "Daniel Deronda"),
        (6688, "The Mill on the Floss"),
        (550, "Silas Marner"),
    ],
    "Jane_Austen": [
        (1342, "Pride and Prejudice"),
        (158, "Emma"),
        (161, "Sense and Sensibility"),
        (141, "Mansfield Park"),
        (121, "Northanger Abbey"),
        (105, "Persuasion"),
    ],
    "Arthur_Conan_Doyle": [
        (1661, "The Adventures of Sherlock Holmes"),
        (834, "The Memoirs of Sherlock Holmes"),
        (108, "The Return of Sherlock Holmes"),
        (244, "A Study in Scarlet"),
        (2097, "The Sign of the Four"),
        (2852, "The Hound of the Baskervilles"),
        (3289, "The Valley of Fear"),
    ],
}


def get_gutenberg_url(book_id: int) -> str:
    """Return the canonical Project Gutenberg raw text URL for a book."""
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def sanitize_filename(name: str) -> str:
    """Convert titles into filesystem-friendly names."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[-\s]+", "_", name).strip("_")
    return name


def strip_headers(text: str) -> str:
    """Remove common Project Gutenberg boilerplate to reduce embedding noise."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines[:500]):
        if any(marker in line for marker in start_markers):
            start_idx = i + 1
            break

    for i, line in enumerate(lines[-500:]):
        if any(marker in line for marker in end_markers):
            end_idx = len(lines) - 500 + i
            break

    return "\n".join(lines[start_idx:end_idx])


def fetch_book(book_id: int, author: str, title: str, output_dir: Path) -> bool:
    """Download, clean, and persist a single book. Returns True on success."""
    url = get_gutenberg_url(book_id)

    try:
        response: Response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to download {} (ID {}): {}", title, book_id, exc)
        return False

    response.encoding = "utf-8"
    cleaned_text = strip_headers(response.text)

    safe_title = sanitize_filename(title)
    safe_author = sanitize_filename(author)
    filename = f"{safe_author}-{safe_title}.txt"
    filepath = output_dir / filename

    if filepath.exists():
        logger.info("Skipping existing file {}", filepath)
        return False

    filepath.write_text(cleaned_text, encoding="utf-8")
    logger.debug("Wrote {} to {}", title, filepath)
    return True


def download_corpus(output_dir: Path, sleep_seconds: float) -> None:
    """Iterate the curated corpus and persist files to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing corpus to {}", output_dir.resolve())

    total_books = sum(len(books) for books in BOOKS.values())
    success_count = 0

    for author, books in BOOKS.items():
        logger.info("--- Processing {} ---", author.replace("_", " "))
        for book_id, title in tqdm(books, desc=f"Downloading {author}", unit="book"):
            if fetch_book(book_id, author, title, output_dir):
                success_count += 1
            time.sleep(sleep_seconds)

    logger.info("=" * 40)
    logger.info(
        "Downloaded {}/{} books into {}", success_count, total_books, output_dir
    )
    logger.info("=" * 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the curated Project Gutenberg corpus into data/raw."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the raw texts (default: data/raw).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Friendly delay between requests in seconds (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_corpus(output_dir=args.output_dir, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()


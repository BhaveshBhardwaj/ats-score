"""
Streaming JSONL data loader.

Reads candidates.jsonl line-by-line to stay within 16GB RAM.
Supports both plain .jsonl and gzipped .jsonl.gz files.
"""

import gzip
import json
from pathlib import Path


def load_candidates(filepath: str, limit: int = None):
    """
    Generator that yields candidate dicts one at a time.
    
    Supports:
        - .jsonl (line-delimited JSON)
        - .jsonl.gz (gzipped line-delimited JSON)
        - .json (JSON array)
    
    Args:
        filepath: Path to candidates file
        limit: Optional max number of candidates to yield (for testing)
    
    Yields:
        dict: A single candidate profile
    """
    path = Path(filepath)
    
    # Handle .json files (JSON array format)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = 0
            for candidate in data:
                yield candidate
                count += 1
                if limit and count >= limit:
                    break
        else:
            yield data
        return
    
    # Handle .jsonl and .jsonl.gz files
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    
    count = 0
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                if isinstance(candidate, list):
                    # Handle case where entire file is a JSON array on one line
                    for c in candidate:
                        yield c
                        count += 1
                        if limit and count >= limit:
                            return
                else:
                    yield candidate
                    count += 1
                    if limit and count >= limit:
                        break
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line: {e}")
                continue


def load_candidates_list(filepath: str, limit: int = None) -> list:
    """
    Load all candidates into memory as a list.
    Use only when you need random access; prefer the generator for streaming.
    
    Args:
        filepath: Path to candidates.jsonl or candidates.jsonl.gz
        limit: Optional max number of candidates to load
    
    Returns:
        list[dict]: All candidate profiles
    """
    return list(load_candidates(filepath, limit=limit))


def count_candidates(filepath: str) -> int:
    """Count total candidates without loading all into memory."""
    path = Path(filepath)
    
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    
    count = 0
    with opener() as f:
        for line in f:
            if line.strip():
                count += 1
    return count

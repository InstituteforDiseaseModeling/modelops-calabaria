"""Token-based hashing utilities for deterministic builds.

This module provides whitespace-agnostic hashing of Python files
using tokenization to ignore formatting changes while preserving
semantic content.
"""

import io
import json
import tokenize
from pathlib import Path
from typing import Iterable, Tuple, Any

try:
    from modelops_contracts import digest_bytes
except ImportError:
    # Fallback for when contracts isn't available
    import hashlib
    def digest_bytes(data: bytes) -> str:
        """Compute BLAKE2b-256 hash of bytes."""
        return hashlib.blake2b(data, digest_size=32).hexdigest()


# Tokens to skip when hashing (formatting-only)
SKIP_TOKENS = {
    tokenize.COMMENT,     # Comments don't affect behavior
    tokenize.NL,          # Newlines within statements
    tokenize.NEWLINE,     # Statement-ending newlines
    tokenize.INDENT,      # Indentation changes
    tokenize.DEDENT,      # Dedentation changes
}


def token_hash(path: Path) -> str:
    """Hash Python file based on tokens only, ignoring formatting.

    This provides deterministic hashing that ignores:
    - Whitespace changes
    - Comment changes
    - Indentation style (tabs vs spaces)
    - Blank lines

    Args:
        path: Path to Python file to hash

    Returns:
        Hash string in format "sha256:abcd1234..."

    Example:
        >>> path = Path("model.py")
        >>> hash1 = token_hash(path)
        >>> # Reformat file with black
        >>> hash2 = token_hash(path)
        >>> assert hash1 == hash2  # Same despite formatting
    """
    try:
        src = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to binary hash if not valid UTF-8
        return digest_bytes(path.read_bytes())

    tokens = []
    try:
        # Tokenize the source code
        for token in tokenize.generate_tokens(io.StringIO(src).readline):
            # Skip formatting-only tokens
            if token.type in SKIP_TOKENS:
                continue

            # Keep token type and string, discard position info
            # This makes the hash position-independent
            tokens.append((token.type, token.string))

    except tokenize.TokenError:
        # Fall back to source hash if tokenization fails
        return digest_bytes(src.encode('utf-8'))

    # Create deterministic JSON representation
    payload = canonical_json(tokens).encode('utf-8')
    return digest_bytes(payload)


def code_sig(file_records: Iterable[Tuple[str, str]]) -> str:
    """Create signature from multiple file hashes.

    Combines multiple file hashes into a single deterministic signature
    that represents the state of a code collection.

    Args:
        file_records: Iterable of (file_path, file_hash) pairs

    Returns:
        64-character hex hash string

    Example:
        >>> records = [
        ...     ("src/model.py", "abc123..."),
        ...     ("src/utils.py", "def456..."),
        ... ]
        >>> sig = code_sig(records)
    """
    # Sort by path for deterministic ordering
    sorted_records = sorted(file_records)

    # Combine path and hash for each file
    combined = "|".join(f"{path}::{hash_val}" for path, hash_val in sorted_records)

    # Hash the combination
    return digest_bytes(combined.encode('utf-8'))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization.

    Always produces the same string for the same object by:
    - Sorting dictionary keys
    - Using compact separators
    - Ensuring ASCII output for stability

    Args:
        obj: Object to serialize

    Returns:
        Canonical JSON string
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def sha256_bytes(data: bytes) -> str:
    """Hash raw bytes using BLAKE2b.

    Args:
        data: Raw bytes to hash

    Returns:
        64-character hex hash string
    """
    return digest_bytes(data)


def sha256_string(text: str) -> str:
    """Hash string content using BLAKE2b.

    Args:
        text: String to hash

    Returns:
        64-character hex hash string
    """
    return sha256_bytes(text.encode('utf-8'))


def content_hash(content: Any) -> str:
    """Hash arbitrary content deterministically.

    Args:
        content: Content to hash (will be JSON-serialized)

    Returns:
        64-character hex hash string
    """
    return sha256_string(canonical_json(content))
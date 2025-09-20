"""Tests for CLI hashing utilities.

Tests token-based file hashing, content hashing, and code signatures
for deterministic manifest generation.
"""

import pytest
from pathlib import Path
import tempfile
import textwrap

from modelops_calabaria.cli.hashing import (
    token_hash,
    content_hash,
    code_sig,
    canonical_json,
    sha256_bytes,
)


class TestTokenHashing:
    """Tests for token-based file hashing."""

    def test_token_hash_ignores_whitespace(self):
        """Token hash should be identical despite different whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create two files with identical tokens but different formatting
            file1 = tmpdir / "file1.py"
            file2 = tmpdir / "file2.py"

            file1.write_text(textwrap.dedent("""
                def hello(name):
                    return f"Hello, {name}!"

                if __name__ == "__main__":
                    print(hello("world"))
            """).strip())

            file2.write_text(textwrap.dedent("""
                def hello( name ):
                    return f"Hello, {name}!"

                if __name__=="__main__":
                    print( hello( "world" ) )
            """).strip())

            assert token_hash(file1) == token_hash(file2)

    def test_token_hash_ignores_comments(self):
        """Token hash should be identical despite different comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            file1 = tmpdir / "file1.py"
            file2 = tmpdir / "file2.py"

            file1.write_text("def hello():\n    return 42")
            file2.write_text("def hello():\n    # This is a comment\n    return 42")

            assert token_hash(file1) == token_hash(file2)

    def test_token_hash_detects_semantic_changes(self):
        """Token hash should differ when semantic content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            file1 = tmpdir / "file1.py"
            file2 = tmpdir / "file2.py"

            file1.write_text("def hello(): return 42")
            file2.write_text("def hello(): return 43")  # Different value

            assert token_hash(file1) != token_hash(file2)

    def test_token_hash_detects_name_changes(self):
        """Token hash should differ when names change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            file1 = tmpdir / "file1.py"
            file2 = tmpdir / "file2.py"

            file1.write_text("def hello(): return 42")
            file2.write_text("def goodbye(): return 42")  # Different name

            assert token_hash(file1) != token_hash(file2)

    def test_token_hash_syntax_error(self):
        """Token hash should handle syntax errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            bad_file = tmpdir / "bad.py"
            bad_file.write_text("def hello(:\n    return 42")  # Syntax error

            # Should not raise an exception, should return consistent hash
            hash1 = token_hash(bad_file)
            hash2 = token_hash(bad_file)
            assert hash1 == hash2
            assert len(hash1) == 64  # Plain hex without prefix


class TestContentHashing:
    """Tests for content-based hashing."""

    def test_content_hash_string(self):
        """Content hash should work for string inputs."""
        hash1 = content_hash("hello world")
        hash2 = content_hash("hello world")
        hash3 = content_hash("hello world!")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 71  # "sha256:" + 64 hex chars

    def test_content_hash_dict(self):
        """Content hash should work for dictionary inputs."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}  # Different order
        dict3 = {"a": 1, "b": 3}  # Different value

        hash1 = content_hash(dict1)
        hash2 = content_hash(dict2)
        hash3 = content_hash(dict3)

        assert hash1 == hash2  # Order should not matter
        assert hash1 != hash3  # Content should matter


class TestCodeSig:
    """Tests for code signature generation."""

    def test_code_sig_empty(self):
        """Code signature should work with empty file list."""
        sig = code_sig([])
        assert len(sig) == 64  # SHA256 hex length

    def test_code_sig_deterministic(self):
        """Code signature should be deterministic."""
        files = [("path1", "hash1"), ("path2", "hash2")]
        sig1 = code_sig(files)
        sig2 = code_sig(files)
        assert sig1 == sig2

    def test_code_sig_order_independent(self):
        """Code signature should be order-independent."""
        files1 = [("path1", "hash1"), ("path2", "hash2")]
        files2 = [("path2", "hash2"), ("path1", "hash1")]

        sig1 = code_sig(files1)
        sig2 = code_sig(files2)
        assert sig1 == sig2

    def test_code_sig_content_dependent(self):
        """Code signature should depend on file content."""
        files1 = [("path1", "hash1"), ("path2", "hash2")]
        files2 = [("path1", "hash1"), ("path2", "hash3")]  # Different hash

        sig1 = code_sig(files1)
        sig2 = code_sig(files2)
        assert sig1 != sig2


class TestCanonicalJson:
    """Tests for canonical JSON serialization."""

    def test_canonical_json_sorts_keys(self):
        """Canonical JSON should sort keys deterministically."""
        data = {"z": 1, "a": 2, "m": 3}
        json_str = canonical_json(data)

        # Should be sorted by keys
        assert json_str == '{"a":2,"m":3,"z":1}'

    def test_canonical_json_nested(self):
        """Canonical JSON should sort nested objects too."""
        data = {
            "outer": {"z": 1, "a": 2},
            "array": [{"b": 2, "a": 1}]
        }
        json_str = canonical_json(data)

        # All keys should be sorted
        expected = '{"array":[{"a":1,"b":2}],"outer":{"a":2,"z":1}}'
        assert json_str == expected

    def test_canonical_json_reproducible(self):
        """Canonical JSON should always produce same output."""
        data = {"z": 1, "a": 2, "nested": {"y": 3, "x": 4}}

        json1 = canonical_json(data)
        json2 = canonical_json(data)

        assert json1 == json2


class TestSha256Bytes:
    """Tests for raw bytes hashing."""

    def test_sha256_bytes_basic(self):
        """SHA256 should work for basic byte inputs."""
        data = b"hello world"
        hash_value = sha256_bytes(data)

        assert len(hash_value) == 71  # "sha256:" + 64 hex chars
        assert hash_value.startswith("sha256:")

    def test_sha256_bytes_deterministic(self):
        """SHA256 should be deterministic."""
        data = b"test data"
        hash1 = sha256_bytes(data)
        hash2 = sha256_bytes(data)

        assert hash1 == hash2

    def test_sha256_bytes_different_inputs(self):
        """SHA256 should differ for different inputs."""
        hash1 = sha256_bytes(b"data1")
        hash2 = sha256_bytes(b"data2")

        assert hash1 != hash2


class TestIntegration:
    """Integration tests for hashing utilities."""

    def test_real_python_file(self):
        """Test hashing a real Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a realistic Python file
            py_file = tmpdir / "model.py"
            py_file.write_text(textwrap.dedent("""
                '''A simple model for testing.'''

                from dataclasses import dataclass
                from typing import Dict, Any

                @dataclass
                class TestModel:
                    '''Test model class.'''

                    def __init__(self, config: Dict[str, Any]):
                        self.config = config

                    def simulate(self, params):
                        '''Run simulation.'''
                        return {"result": params.get("value", 0) * 2}
            """).strip())

            # Should produce consistent hash
            hash1 = token_hash(py_file)
            hash2 = token_hash(py_file)

            assert hash1 == hash2
            assert len(hash1) == 64

    def test_code_sig_with_real_files(self):
        """Test code signature with multiple real files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple Python files
            file1 = tmpdir / "module1.py"
            file2 = tmpdir / "module2.py"

            file1.write_text("def func1(): return 1")
            file2.write_text("def func2(): return 2")

            # Create file records
            records = [
                (str(file1.relative_to(tmpdir)), token_hash(file1)),
                (str(file2.relative_to(tmpdir)), token_hash(file2)),
            ]

            sig = code_sig(records)
            assert len(sig) == 64

            # Should be reproducible
            sig2 = code_sig(records)
            assert sig == sig2
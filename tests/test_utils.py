"""Tests for utils: parse_drive_url, chunk_text."""

from utils import chunk_text, parse_drive_url


class TestParseDriveUrl:
    def test_none_returns_none(self):
        assert parse_drive_url(None) is None

    def test_empty_string_returns_none(self):
        assert parse_drive_url("") is None
        assert parse_drive_url("   ") is None

    def test_folder_url(self):
        url = "https://drive.google.com/drive/folders/abc123xyz"
        assert parse_drive_url(url) == ("folder", "abc123xyz")

    def test_file_url(self):
        url = "https://drive.google.com/file/d/fileId123/view"
        assert parse_drive_url(url) == ("file", "fileId123")

    def test_docs_url(self):
        url = "https://docs.google.com/document/d/docId456/edit"
        assert parse_drive_url(url) == ("file", "docId456")

    def test_invalid_url_returns_none(self):
        assert parse_drive_url("https://example.com/other") is None
        assert parse_drive_url("not-a-url") is None


class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("hello") == ["hello"]
        assert chunk_text("x" * 500, chunk_size=1000) == ["x" * 500]

    def test_exact_chunk_size(self):
        text = "a" * 1000
        assert chunk_text(text, chunk_size=1000, overlap=0) == [text]

    def test_splits_with_overlap(self):
        text = "a" * 1500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2
        assert all(len(c) <= 1000 for c in chunks)
        # With overlap, chunks overlap so total length can exceed len(text)
        assert sum(len(c) for c in chunks) >= len(text)

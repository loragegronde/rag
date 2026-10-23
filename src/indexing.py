"""Build a BM25 index over the local vLLM corpus."""

from pathlib import Path
from typing import Any

import bm25s
import Stemmer
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from tqdm import tqdm

ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".rst"}





class Index:

    def __init__(
        self,
        max_chunk_size: int,
        raw_dir: Path,
        index_dir: Path,
    ) -> None:
        self.max_chunk_size: int = max_chunk_size
        self.raw_dir: Path = raw_dir
        self.index_dir: Path = index_dir
        self.stemmer: Any = Stemmer.Stemmer("english")
        self.chunk_overlap: int = min(max_chunk_size // 10, max_chunk_size - 1)

    def build(self) -> None:
        file_paths = self._collect_files()
        if not file_paths:
            print(f"No indexable files found in {self.raw_dir}")
            return

        corpus_chunks: list[str] = []
        chunk_metadata: list[dict[str, str | int]] = []

        for file_path in tqdm(file_paths, desc="Chunking", unit="file"):
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Skipping {file_path}: {exc}")
                continue

            for start, end in self._chunk_file(content, file_path.suffix):
                corpus_chunks.append(content[start:end])
                chunk_metadata.append(
                    {
                        "file_path": str(file_path),
                        "first_character_index": start,
                        "last_character_index": end,
                    }
                )

        with tqdm(
            total=len(corpus_chunks),
            desc="Tokenizing",
            unit="chunk",
        ) as progress:
            corpus_tokens = bm25s.tokenize(
                corpus_chunks,
                stemmer=self.stemmer,
                show_progress=False,
            )
            _ = progress.update(len(corpus_chunks))

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(
            str(self.index_dir),
            corpus=chunk_metadata,
            show_progress=True,
        )
        print(
            f"Ingestion complete! Indexed {len(corpus_chunks)} chunks ",
            f"under {self.index_dir}/",
        )

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.raw_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            files.append(path)
        return sorted(files)

    def _chunk_file(self, content: str, suffix: str) -> list[tuple[int, int]]:
        """Split ``content`` and return exact ``[start, end)`` spans."""
        if not content.strip():
            return []

        suffix = suffix.lower()
        if suffix == ".py":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif suffix == ".md":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        spans: list[tuple[int, int]] = []
        cursor = 0
        for chunk in splitter.split_text(content):
            text = chunk.strip("\n")
            if not text:
                continue
            start = content.find(text, cursor)
            if start == -1:
                start = content.find(text)
            if start == -1:
                continue
            end = start + len(text)
            spans.append((start, end))
            cursor = start + 1
        return spans

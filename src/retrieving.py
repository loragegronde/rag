"""BM25 retrieval over a persisted index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
import Stemmer

import bm25s
from tqdm import tqdm

from .models import MinimalSearchResults, MinimalSource, StudentSearchResults


class Retriever:
    """Load a saved BM25 index and answer single or batch queries."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir: Path = index_dir
        self.retriever: bm25s.BM25 = bm25s.BM25.load(
            str(index_dir), load_corpus=True
        )
        self.stemmer: Any = Stemmer.Stemmer("english")

    def search_single(
        self,
        query: str,
        k: int,
        question_id: str = "q1",
    ) -> MinimalSearchResults:
        """Return the top-k sources for one query."""
        query_tokens = bm25s.tokenize(
            [query],
            stemmer=self.stemmer,
            show_progress=False,
        )
        results, _ = cast(
            tuple[Any, Any],
            self.retriever.retrieve(query_tokens, k=k, show_progress=False),
        )
        sources: list[MinimalSource] = [
            self._chunk_to_source(chunk) for chunk in results[0]
        ]
        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=sources,
        )

    def search_dataset(
        self,
        dataset_path: Path,
        save_directory: Path,
        k: int,
    ) -> Path:
        """Search every question in a dataset."""
        with dataset_path.open("r", encoding="utf-8") as handle:
            dataset: dict[str, Any] = json.load(handle)

        questions_data: list[dict[str, Any]] = cast(
            list[dict[str, Any]],
            dataset.get("rag_questions", []),
        )
        queries: list[str] = [str(item["question"]) for item in questions_data]
        query_ids: list[str] = [
            str(item.get("question_id", index))
            for index, item in enumerate(questions_data)
        ]

        queries_tokens = bm25s.tokenize(
            queries,
            stemmer=self.stemmer,
            show_progress=False,
        )

        results, _ = cast(
            tuple[Any, Any],
            self.retriever.retrieve(queries_tokens, k=k, show_progress=False),
        )

        all_search_results: list[MinimalSearchResults] = []
        for index in tqdm(range(len(queries)), desc="Formatting results"):
            sources: list[MinimalSource] = [
                self._chunk_to_source(chunk) for chunk in results[index]
            ]
            all_search_results.append(
                MinimalSearchResults(
                    question_id=query_ids[index],
                    question=queries[index],
                    retrieved_sources=sources,
                )
            )

        final_output = StudentSearchResults(
            search_results=all_search_results,
            k=k,
        )
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path: Path = save_directory / dataset_path.name
        _ = save_path.write_text(
            final_output.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return save_path

    @staticmethod
    def _chunk_to_source(chunk: Any) -> MinimalSource:
        return MinimalSource(
            file_path=str(chunk["file_path"]),
            first_character_index=int(chunk["first_character_index"]),
            last_character_index=int(chunk["last_character_index"]),
        )

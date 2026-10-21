from pathlib import Path

from src.retrieving import Retriever
from src.indexing import Index


class Cli:
    """Commands available through the command-line interface."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_directory: str = "data/raw",
        index_directory: str = "data/processed",
    ) -> None:
        """Build an index from raw documents."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        raw_dir = self._validate_directory(raw_directory, create=False)
        index_dir = self._validate_directory(index_directory, create=True)

        Index(max_chunk_size, raw_dir, index_dir).build()

    def search(
        self,
        query: str,
        index_directory: str = "data/processed",
        k: int = 10,
    ) -> None:
        """Return the top-k sources for a query."""
        self._validate_query(query, k)
        index_dir = self._validate_directory(index_directory, create=False)

        retriever = Retriever(index_dir)
        result = retriever.search_single(query, k)
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        index_directory: str = "data/processed",
        k: int = 10,
    ) -> None:
        """Search every query in a dataset and save the results."""

        self._validate_k(k)

        dataset_file: Path = self._validate_path(dataset_path)
        index_dir: Path = self._validate_directory(
            index_directory, create=False
        )
        save_dir: Path = self._validate_directory(save_directory, create=True)

    def answer(
        self,
        query: str,
        index_directory: str = "data/processed",
        k: int = 10,
    ) -> None:
        """Answer a query using retrieved sources."""
        self._validate_query(query, k)
        index_dir: Path = self._validate_directory(
            index_directory, create=False
        )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        index_directory: str = "data/processed",
    ) -> None:
        """Generate answers for previously retrieved dataset results."""
        results_file: Path = self._validate_path(student_search_results_path)
        index_dir: Path = self._validate_directory(
            index_directory, create=False
        )
        save_dir: Path = self._validate_directory(save_directory, create=True)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality using recall@k."""
        results_file: Path = self._validate_path(student_search_results_path)
        dataset_file: Path = self._validate_path(dataset_path)

    @staticmethod
    def _validate_directory(path: str, *, create: bool) -> Path:
        directory = Path(path)
        if directory.is_dir():
            return directory
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        raise FileNotFoundError(f"Directory not found: {directory}")

    @staticmethod
    def _validate_path(path: str) -> Path:
        file_path = Path(path)
        if file_path.is_file():
            return file_path
        raise FileNotFoundError(f"File not found: {file_path}")

    @staticmethod
    def _validate_k(k: int) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

    @staticmethod
    def _validate_query(query: str, k: int) -> None:
        Cli._validate_k(k)
        if not query.strip():
            raise ValueError("Query cannot be empty")

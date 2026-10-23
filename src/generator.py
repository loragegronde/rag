
from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 256
SNIPPET_SEPARATOR = "\n\n---\n\n"
THINK_END_TAG = "</think>"
INSTRUCTIONS = (
    "Answer using ONLY the context below. "
    "Quote exact code expressions or literals; do not invent numbers. "
    "If defaults are expressions (e.g. float8_info.min), report those "
    "expressions exactly."
)


def select_device() -> torch.device:
    """Pick the fastest available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Generator:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.max_context_chars: int = max_context_chars
        self.max_new_tokens: int = max_new_tokens
        self.device: torch.device = select_device()

        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_name),
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        )
        _ = torch.nn.Module.to(self.model, self.device)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Answer one question from its retrieved sources."""
        return self._generate(self._build_prompt(question, sources))

    def answer_result(self, result: MinimalSearchResults) -> MinimalAnswer:
        """Attach a generated answer to a search result."""
        return MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=self.answer(result.question, result.retrieved_sources),
        )

    def answer_dataset(
        self,
        search_results_path: Path,
        save_directory: Path,
    ) -> Path:
        """Answer every search result in a file and save the output."""
        search_results = StudentSearchResults.model_validate_json(
            search_results_path.read_text(encoding="utf-8")
        )
        answers = [
            self.answer_result(result)
            for result in tqdm(
                search_results.search_results, desc="Generating answers"
            )
        ]
        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )

        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / search_results_path.name
        _ = save_path.write_text(
            output.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return save_path

    def _build_prompt(
        self, question: str, sources: list[MinimalSource]
    ) -> str:
        context = self._build_context(question, sources)
        content = (
            f"{INSTRUCTIONS}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return cast(str, prompt)

    def _build_context(
        self, question: str, sources: list[MinimalSource]
    ) -> str:
        markers = self._query_markers(question)
        scored: list[tuple[int, str]] = []
        for source in sources:
            text = self._extract_text(source)
            if not text:
                continue
            overlap = sum(marker in text for marker in markers)
            header = (
                f"Source: {source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )
            scored.append((overlap, f"{header}\n{text}"))
        scored.sort(key=lambda item: item[0], reverse=True)

        blocks: list[str] = []
        remaining = self.max_context_chars
        for _, block in scored:
            if remaining <= 0:
                break
            block = block[:remaining]
            blocks.append(block)
            remaining -= len(block) + len(SNIPPET_SEPARATOR)
        return SNIPPET_SEPARATOR.join(blocks)

    @staticmethod
    def _query_markers(question: str) -> list[str]:
        cleaned = question.replace("'", " ").replace('"', " ")
        return [
            token
            for token in cleaned.split()
            if "_" in token or any(char.isupper() for char in token)
        ]

    @staticmethod
    def _extract_text(source: MinimalSource) -> str:
        start = source.first_character_index
        length = source.last_character_index - start
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                _ = f.read(start)
                return f.read(length)
        except (OSError, UnicodeDecodeError) as e:
            print(f"Error reading {source.file_path}: {e}")
            return ""

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = cast(
                torch.Tensor,
                self.model.generate( 
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                ),
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        text = cast(
            str,
            self.tokenizer.decode(
                outputs[0][prompt_length:], skip_special_tokens=True
            ),
        )
        return text.split(THINK_END_TAG)[-1].strip()

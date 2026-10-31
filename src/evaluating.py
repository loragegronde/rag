"""Recall@k evaluation against a ground-truth answered dataset."""

from __future__ import annotations

from pathlib import Path

from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

DEFAULT_IOU_THRESHOLD = 0.05
DEFAULT_K_VALUES = (1, 3, 5, 10)


def compare_sources(s1: MinimalSource, s2: MinimalSource) -> float:
    """IoU of two character spans; 0.0 when they are in different files."""
    if s1.file_path != s2.file_path:
        return 0.0

    s1_len = s1.last_character_index - s1.first_character_index
    s2_len = s2.last_character_index - s2.first_character_index
    intersection = max(
        0,
        min(s1.last_character_index, s2.last_character_index)
        - max(s1.first_character_index, s2.first_character_index),
    )
    union = s1_len + s2_len - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def recall_at_k_for_question(
    pred_sources: list[MinimalSource],
    true_sources: list[MinimalSource],
    *,
    minimal_iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Fraction of ground-truth sources found in the predictions.

    A true source counts as found when any prediction overlaps it with
    IoU greater than ``minimal_iou_threshold``. Questions with no ground
    truth return 1.0; questions with no predictions return 0.0.
    """
    if not true_sources:
        return 1.0
    if not pred_sources:
        return 0.0

    found = 0
    for true_source in true_sources:
        if any(
            compare_sources(true_source, pred) > minimal_iou_threshold
            for pred in pred_sources
        ):
            found += 1
    return found / len(true_sources)


def evaluate_recall(
    student_search_results_path: Path,
    dataset_path: Path,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    minimal_iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    """Average recall@k over answered questions; print a short report."""
    student = StudentSearchResults.model_validate_json(
        student_search_results_path.read_text(encoding="utf-8")
    )
    dataset = RagDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )

    answered = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    ]
    preds_by_id = {
        result.question_id: result.retrieved_sources
        for result in student.search_results
    }

    print(f"Total number of questions: {len(dataset.rag_questions)}")
    print(f"Total number of questions with sources: {len(answered)}")
    with_preds = sum(1 for q in answered if preds_by_id.get(q.question_id))
    print(f"Total number of questions with student sources: {with_preds}")

    averages: dict[str, float] = {}
    for k in k_values:
        scores = [
            recall_at_k_for_question(
                preds_by_id.get(question.question_id, [])[:k],
                question.sources,
                minimal_iou_threshold=minimal_iou_threshold,
            )
            for question in answered
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        averages[f"recall@{k}"] = avg

    print("\nEvaluation Results")
    print("========================================")
    print(f"Questions evaluated: {len(answered)}")
    for k in k_values:
        recall = averages[f"recall@{k}"]
        print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")
    print(averages)
    return averages

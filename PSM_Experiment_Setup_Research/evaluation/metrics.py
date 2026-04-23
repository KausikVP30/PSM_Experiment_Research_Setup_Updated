from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    for gold in gold_answers:
        if pred == normalize_answer(gold):
            return 1.0
    return 0.0


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            score = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def rouge_l(prediction: str, gold_answers: Sequence[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = 0.0
    for gold in gold_answers:
        score = scorer.score(gold, prediction)["rougeL"].fmeasure
        best = max(best, score)
    return best


def bleu(prediction: str, gold_answers: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    refs = [normalize_answer(g).split() for g in gold_answers if normalize_answer(g)]
    if not refs:
        return 0.0

    smoothie = SmoothingFunction().method1
    return float(sentence_bleu(refs, pred_tokens, smoothing_function=smoothie))


def contains_any_answer(text: str, gold_answers: Sequence[str]) -> bool:
    norm_text = normalize_answer(text)
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold and norm_gold in norm_text:
            return True
    return False

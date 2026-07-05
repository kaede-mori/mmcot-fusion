"""Evaluation metrics for ScienceQA answer accuracy and rationale quality.

Adapted from https://github.com/amazon-science/mm-cot (Apache-2.0)
(``utils_evaluate.py`` / ``evaluations.py``), made device-agnostic; the
sentence-similarity metric is optional. Adds McNemar's test for paired
comparison of two systems on the same test questions.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

ANSWER_PATTERN = re.compile(r"The answer is \(([A-Z])\)")


def extract_answer(text):
    res = ANSWER_PATTERN.findall(text)
    return res[0] if len(res) == 1 else "FAILED"


def _tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def calculate_bleu(results, references, gram):
    bleus = []
    weights = tuple([1.0 / gram] * gram)
    for qid, output in results.items():
        prediction = _tokenize(output)
        target = _tokenize(references[qid])
        if len(prediction) < gram:
            bleus.append(0.0)
            continue
        bleus.append(sentence_bleu([target], prediction, weights=weights))
    return float(np.mean(bleus))


def calculate_rouge(results, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(references[qid], output)["rougeL"].fmeasure
        for qid, output in results.items()
    ]
    return float(np.mean(scores))


def calculate_similarity(results, references, model):
    scores = []
    for qid, output in results.items():
        emb = model.encode([output, references[qid]], convert_to_tensor=True)
        scores.append(float((emb[0] @ emb[1]) / (emb[0].norm() * emb[1].norm())))
    return float(np.mean(scores))


def get_acc_with_condition(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd["true_false"] == True]  # noqa: E712
    return "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)


def get_scores(result_data, rationale_data, results_reference, data_file, similarity_model=None):
    results = result_data
    num = len(results)

    sqa_data = json.load(open(data_file))
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd["split"] == "test"]

    for index, row in res_pd.iterrows():
        res_pd.loc[index, "no_context"] = not row["hint"] and not row["image"]
        res_pd.loc[index, "has_text"] = bool(row["hint"])
        res_pd.loc[index, "has_image"] = bool(row["image"])
        res_pd.loc[index, "has_text_image"] = bool(row["hint"] and row["image"])
        label = row["answer"]
        pred = int(results[index])
        res_pd.loc[index, "pred"] = pred
        res_pd.loc[index, "true_false"] = label == pred

    acc_average = len(res_pd[res_pd["true_false"] == True]) / num * 100  # noqa: E712

    scores = {
        "answer": {
            "acc_natural": get_acc_with_condition(res_pd, "subject", "natural science"),
            "acc_social": get_acc_with_condition(res_pd, "subject", "social science"),
            "acc_language": get_acc_with_condition(res_pd, "subject", "language science"),
            "acc_has_text": get_acc_with_condition(res_pd, "has_text", True),
            "acc_has_image": get_acc_with_condition(res_pd, "has_image", True),
            "acc_no_context": get_acc_with_condition(res_pd, "no_context", True),
            "acc_grade_1_6": get_acc_with_condition(
                res_pd, "grade", [f"grade{i}" for i in range(1, 7)]
            ),
            "acc_grade_7_12": get_acc_with_condition(
                res_pd, "grade", [f"grade{i}" for i in range(7, 13)]
            ),
            "acc_average": "{:.2f}".format(acc_average),
        },
        "rationale": {
            "bleu1": calculate_bleu(rationale_data, results_reference, gram=1) * 100,
            "bleu4": calculate_bleu(rationale_data, results_reference, gram=4) * 100,
            "rouge": calculate_rouge(rationale_data, results_reference) * 100,
        },
    }
    if similarity_model is not None:
        scores["rationale"]["similarity"] = (
            calculate_similarity(rationale_data, results_reference, similarity_model) * 100
        )
    return scores


def mcnemar_test(correct_a, correct_b):
    """Exact McNemar's test on paired per-question correctness vectors.

    Returns the discordant counts and the two-sided p-value of an exact
    binomial test on the discordant pairs.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Correctness vectors must have identical shapes")
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n = b + c
    p_value = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p_value)}

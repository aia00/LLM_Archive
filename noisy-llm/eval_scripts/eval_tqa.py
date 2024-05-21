# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_eval_score(prediction, answer_item, mute=True):
    ground_truths = answer_item['normalized_aliases']
    
    # exact match
    em_for_this_question = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    if em_for_this_question == 0 and not mute:
        print(f'EM=0! PREDICTION: {prediction}, Answers: {ground_truths}')
    
    # f1
    f1_for_this_question = metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)
    
    return {'exact_match': em_for_this_question, 'f1': f1_for_this_question}
    
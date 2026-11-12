from shard.loramerge.residence import ResidenceScore, count_passed, score_text


def test_fenced_code_is_not_misclassified_as_word_salad():
    code = """```python
def merge_sorted_lists(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
```"""

    assert score_text(code).verdict == "ok"


def test_low_function_word_prose_still_triggers_word_salad():
    salad = " ".join(f"contentword{i}" for i in range(50))

    assert score_text(salad).verdict == "word_salad"


def test_qwen_residence_summary_counts_only_ok_verdicts():
    scores = {
        "good": ResidenceScore(50, 0.8, 0.01, 0.01, 1, 0.4, "ok"),
        "loop": ResidenceScore(50, 0.2, 0.4, 0.3, 6, 0.3, "repetition"),
        "empty": ResidenceScore(0, 0.0, 0.0, 0.0, 0, 0.0, "empty"),
    }

    assert count_passed(scores) == 1

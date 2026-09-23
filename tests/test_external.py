import pytest
from transformers import AutoTokenizer

from wev.data import labelled_record
from wev.external import single, stratified
from wev.model import encode


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")


def test_single_row_kinds_and_soft_labels_align_with_rendered_options(tok):
    rows = [single({"kind": "choice", "options": ["b", "a", "c"], "target": [0.2, 0.7, 0.1], "state": "s", "question": "q?"}, "t"),
            single({"kind": "noul", "options": ["true", "false"], "target": [0.3, 0.7], "state": "s", "question": "q?"}, "t"),
            single({"kind": "score", "options": ["0", "1", "2"], "target": [0.1, 0.2, 0.7], "state": "s", "question": "q?"}, "t")]
    assert [r["labels"]["q"] for r in rows] == ["a", False, 2]
    softs = []
    for r in rows:
        rec, meta = labelled_record(r)
        enc = encode(tok, rec)
        softs.append([round(x, 3) for x in enc["soft"][0]])
        assert enc["labels"][0] == max(range(len(enc["soft"][0])), key=enc["soft"][0].__getitem__)
    assert softs == [[0.2, 0.7, 0.1], [0.7, 0.3], [0.1, 0.2, 0.7]]   # choice keeps key order; noul renders [no, yes]


def test_bad_rows_are_dropped():
    assert single({"kind": "choice", "options": ["a", "a"], "target": [0.5, 0.5], "state": "s", "question": "q"}, "t") is None
    assert single({"kind": "noul", "options": ["maybe", "sure"], "target": [0.5, 0.5], "state": "s", "question": "q"}, "t") is None


def test_stratified_spreads_across_groups():
    import random
    groups = ["a"] * 100 + ["b"] * 5 + ["c"] * 50
    picked = stratified(groups, 30, random.Random(0))
    counts = {g: sum(groups[i] == g for i in picked) for g in "abc"}
    assert counts == {"a": 13, "b": 5, "c": 12} or (counts["b"] == 5 and abs(counts["a"] - counts["c"]) <= 1)

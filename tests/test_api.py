import math
import random

from wev.api import to_answers


def test_choice_answers_pass_jev_ultrafast_validation():
    """Mirror of jev_ultrafast.model.validate_choice: keys match, sum within 0.02, choice is the argmax."""
    rng = random.Random(0)
    for k in (2, 8, 60, 200, 255):
        w = [rng.random() ** 3 for _ in range(k)]
        p = [x / sum(w) for x in w]
        keys = [str(i + 1) for i in range(k)]
        a = to_answers([p], [{"id": "q", "type": "choice", "keys": keys}])["q"]
        probs = a["probabilities"]
        assert set(probs) == set(keys)
        assert all(0 <= v <= 1 and math.isfinite(v) for v in [*probs.values(), a["confidence"]])
        assert abs(sum(probs.values()) - 1) < 0.02
        assert probs[a["choice"]] >= max(probs.values()) - 1e-6

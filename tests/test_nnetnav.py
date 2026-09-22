import random

from wev.api import SystemOneRequest
from wev.data import labelled_record
from wev.nnetnav import convert_row, parse_tree

TREE = """RootWebArea 'Wolfram|Alpha: Computational Intelligence', focused, url='https://www.wolframalpha.com/'
\t[51] banner '', visible
\t\t[55] button 'UPGRADE TO PRO', clickable, visible, hasPopup='menu', expanded=False
\t\t\tStaticText 'UPGRADE TO PRO'
\t\t[63] link 'TOUR', clickable, visible, url='https://www.wolframalpha.com/tour'
\t\t\tStaticText 'TOUR'
\t\t[66] button 'Sign in', clickable, visible
\t[74] main '', visible
\t\tStaticText 'Enter what you want to calculate or know about'
\t\t[89] textbox 'WolframAlpha input field', clickable, visible, focused
\t\t[91] button 'Compute input button', clickable, visible
\t\t\t[92] image '', visible
\t\t[108] link 'EXAMPLES', clickable, visible, url='https://www.wolframalpha.com/examples'
\t\t\t[109] image '', visible
\t\t\tStaticText 'EXAMPLES'"""


def prompt(history="1: None"):
    return (f"<|start_header_id|>user<|end_header_id|>\nOBSERVATION:\n{TREE}\nURL: https://www.wolframalpha.com/\n"
            f"OBJECTIVE: Find the chemistry of water.\nPREVIOUS ACTIONS:\n{history}\n<|eot_id|>")


def convert(action, history="1: None"):
    row = {"id": "x1", "task_name": "t1", "prompt": prompt(history),
           "output": f"Let's think. In summary, the next action I will perform is ```{action}``` <|eot_id|>"}
    return convert_row(row, random.Random(0), 10, 10, 1500, 6000)


def target(row):
    op = row["labels"]["operation"]
    q = op.lower() + "_target"
    return row["request"]["questions"][q]["criteria"][row["labels"][q]]["element"]


def test_tree_parsing():
    title, url, nodes, lines = parse_tree(TREE)
    assert title.startswith("Wolfram") and url == "https://www.wolframalpha.com/"
    assert nodes["89"].role == "textbox" and nodes["89"].name == "WolframAlpha input field"
    assert nodes["92"].parent is nodes["91"]
    assert "Enter what you want to calculate or know about" in lines


def test_type_and_click_targets():
    row, why = convert("type [89] [water chemistry] [1]")
    assert why is None and row["labels"]["operation"] == "TYPE_TEXT"
    assert target(row).endswith("WolframAlpha input field")
    row, _ = convert("click [108]")
    assert row["labels"]["operation"] == "CLICK" and target(row).endswith("] EXAMPLES")
    labels = [e["label"] for e in row["request"]["state"]["elements"]]
    assert "image" not in labels          # descendants of the target are never negatives


def test_stop_and_scroll_have_no_target():
    for action, op in [("stop [Water is H2O, molar mass 18.015 g/mol]", "DONE"), ("stop [N/A]", "BLOCKED"),
                       ("scroll [down]", "SCROLL_DOWN")]:
        row, why = convert(action)
        assert why is None and row["labels"] == {"operation": op}
        assert op in row["request"]["questions"]["operation"]["criteria"]


def test_scroll_up_offered_only_after_scrolling_down():
    row, _ = convert("click [108]")
    assert "SCROLL_UP" not in row["request"]["questions"]["operation"]["criteria"]
    row, _ = convert("click [108]", history="1: None\n2: scroll [down]")
    assert "SCROLL_UP" in row["request"]["questions"]["operation"]["criteria"]
    assert row["request"]["state"]["recent_actions"][-1]["kind"] == "scroll"


def test_history_labels():
    row, _ = convert("click [91]", history="1: None\n2: type [89] [water chemistry ] where [89] is WolframAlpha input field")
    assert row["request"]["state"]["recent_actions"] == [
        {"action": "WolframAlpha input field", "kind": "fill", "text": "water chemistry", "page_changed": None}]


def test_unsupported_or_missing_actions_are_dropped():
    assert convert("goto [https://example.com]")[1] == "no_action"
    assert convert("click [99999]")[1] == "target_not_in_tree"


def test_rows_validate_and_render():
    for action in ["type [89] [water] [1]", "click [66]", "stop [done]", "scroll [down]"]:
        row, _ = convert(action)
        SystemOneRequest.model_validate(row["request"])
        rec, meta = labelled_record(row)
        for q, m in zip(rec["questions"], meta):
            assert m["keys"][q["label"]] == row["labels"][m["id"]]

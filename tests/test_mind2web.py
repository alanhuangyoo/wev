import json
import random

from wev.api import SystemOneRequest
from wev.data import labelled_record
from wev.mind2web import Page, convert_step

HTML = """<html backend_node_id="1">
 <body backend_node_id="2">
  <div backend_node_id="3">
   <label backend_node_id="4"><text backend_node_id="5">Reservation type</text></label>
   <select backend_node_id="6" name="type">
     <option backend_node_id="7" value="dine" option_selected="true"><text backend_node_id="8">Dine in</text></option>
     <option backend_node_id="9" value="pickup"><text backend_node_id="10">Pickup</text></option>
   </select>
   <input backend_node_id="11" type="search" placeholder="Find a location"/>
   <a backend_node_id="12" aria_label="Tock home page"/>
   <button backend_node_id="13"><span backend_node_id="14"><text backend_node_id="15">Search</text></span></button>
   <a backend_node_id="16"><text backend_node_id="17">Sign in</text></a>
  </div>
 </body>
</html>"""


def cand(nid, tag, h=20):
    return {"tag": tag, "backend_node_id": nid,
            "attributes": json.dumps({"backend_node_id": nid, "bounding_box_rect": f"0,0,100,{h}"})}


NEG = {"3": cand("3", "div", h=3000), "6": cand("6", "select"), "11": cand("11", "input"), "12": cand("12", "a"),
       "13": cand("13", "button"), "16": cand("16", "a")}


def step(op, value, pos, negs):
    return {"action_uid": f"uid-{pos}", "cleaned_html": HTML, "operation": {"op": op, "original_op": op, "value": value},
            "pos_candidates": [cand(pos, "x")], "neg_candidates": [NEG[n] for n in negs]}


TASK = {"annotation_id": "a1", "website": "exploretock", "domain": "Travel", "subdomain": "Restaurant",
        "confirmed_task": "Find pickup in Boston",
        "action_reprs": ["[combobox]  Reservation type -> SELECT: Pickup",
                         "[searchbox]  Find a location -> TYPE: Boston", "[span]  Search -> CLICK"],
        "actions": [step("SELECT", "Pickup", "6", ["3", "11", "12", "13", "16"]),
                    step("TYPE", "Boston", "11", ["6", "12", "13", "16"]),
                    step("CLICK", "", "14", ["13", "12", "16", "11"])]}


def convert(i):
    row, why = convert_step(TASK, i, Page(HTML), random.Random(0), 10, 10, 1500, 6000)
    assert why is None, why
    SystemOneRequest.model_validate(row["request"])
    return row


def target_of(row):
    op = row["labels"]["operation"]
    qid = op.lower() + "_target"
    return row["request"]["questions"][qid]["criteria"][row["labels"][qid]]


def test_select_step_points_at_the_requested_option():
    row = convert(0)
    assert row["labels"]["operation"] == "SELECT"
    assert target_of(row)["element"].endswith("Reservation type → Pickup")
    labels = [e["label"] for e in row["request"]["state"]["elements"]]
    assert "Reservation type" in labels          # label comes from the preceding <label> text
    assert not any(l.startswith("Reservation type Dine") for l in labels)   # page-sized container (h=3000) dropped
    assert "Reservation type" in row["request"]["state"]["page"]["text"]
    assert row["request"]["state"]["recent_actions"] == []


def test_type_step_and_history():
    row = convert(1)
    assert row["labels"]["operation"] == "TYPE_TEXT"
    assert target_of(row)["element"].endswith("Find a location")
    hist = row["request"]["state"]["recent_actions"]
    assert hist == [{"action": "Reservation type → Pickup", "kind": "select", "text": None, "page_changed": None}]
    # an editable field's click target is labelled like jev-ultrafast's: "Open <label>"
    assert "type_text_target" in row["request"]["questions"] and "click_target" not in row["request"]["questions"]


def test_click_step_excludes_ancestor_of_the_target():
    row = convert(2)
    assert row["labels"]["operation"] == "CLICK"
    assert target_of(row)["element"].endswith("] Search")
    roles = [e["role"] for e in row["request"]["state"]["elements"]]
    assert "button" not in roles                 # the <button> wrapping the gold <span> would also be correct
    assert len(row["request"]["state"]["recent_actions"]) == 2


def test_labels_become_indices_through_the_serving_renderer():
    for i in range(3):
        row = convert(i)
        rec, meta = labelled_record(row)
        assert [m["id"] for m in meta] == list(row["labels"])
        for q, m in zip(rec["questions"], meta):
            assert m["keys"][q["label"]] == row["labels"][m["id"]]
            assert len(q["options"]) == len(m["keys"])

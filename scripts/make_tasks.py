"""Browser tasks for teacher data collection and end-to-end evaluation -> tasks.jsonl {id, url, goal, source, split}.

  nnetnav     first steps of NNetNav-live trajectories (objective + start URL), train from its train split,
              eval from its test split. Tasks that could have real-world effects are dropped.
  flights     Google Flights searches (search only, never booking), random city pairs and future dates.
  wikipedia   open the article about a topic from the Wikipedia main page.
Eval tasks never share a city pair or topic with training tasks.
"""
import argparse
import json
import random
import re

UNSAFE = re.compile(r"\b(buy|purchas|order|book(?!s\b)|reserv|sign ?(in|up)|log ?in|login|register|subscri|post(?!al)|"
                    r"comment|repl(y|ies)|send|email|message|pay|checkout|cart|donat|apply|upload|delete|remove|"
                    r"save|follow|like|share|rate|review|account|password|download|install)", re.I)
CITIES = ["Zurich", "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam", "Vienna", "Lisbon", "Dublin", "Prague",
          "Copenhagen", "Stockholm", "Oslo", "Helsinki", "Warsaw", "Budapest", "Athens", "Istanbul", "Brussels", "Munich",
          "Milan", "Barcelona", "New York", "Chicago", "Boston", "Seattle", "San Francisco", "Los Angeles", "Toronto",
          "Tokyo", "Seoul", "Singapore", "Bangkok", "Sydney", "Dubai", "Hong Kong", "Taipei", "Shanghai", "Beijing"]
TOPICS = ["Gödel's incompleteness theorems", "the Pythagorean theorem", "photosynthesis", "the French Revolution",
          "black holes", "the Great Wall of China", "Alan Turing", "plate tectonics", "the Roman Empire", "DNA",
          "the Industrial Revolution", "quantum entanglement", "the Eiffel Tower", "Ada Lovelace", "the Amazon River",
          "machine learning", "the Renaissance", "Marie Curie", "the Moon landing", "the periodic table",
          "Leonardo da Vinci", "the Pacific Ocean", "the printing press", "Isaac Newton", "the Silk Road",
          "the theory of relativity", "Mount Everest", "the Byzantine Empire", "Charles Darwin", "the Internet",
          "the Sahara", "Beethoven", "the Cold War", "volcanoes", "the human heart", "Nikola Tesla", "the Nile",
          "the Olympic Games", "Shakespeare", "the Big Bang"]
MONTHS = [("October", 2026), ("November", 2026), ("December", 2026), ("January", 2027), ("February", 2027)]


def nnetnav_tasks(path, split):
    seen, out = set(), []
    for line in open(path):
        r = json.loads(line)
        st = r["request"]["state"]
        if st["recent_actions"]:
            continue
        goal = r["request"]["questions"]["operation"]["instructions"]["goal"].strip()
        url = st["page"]["url"]
        if not url.startswith("http") or len(url) > 300 or goal.lower() in seen or UNSAFE.search(goal):
            continue
        seen.add(goal.lower())
        out.append({"url": url, "goal": goal, "source": "nnetnav", "split": split})
    return out


def flights(rng, pairs, split):
    out = []
    for a, b in pairs:
        month, year = rng.choice(MONTHS)
        day = rng.randint(1, 28)
        if rng.random() < 0.6:
            goal = (f"Find one-way flights from {a} to {b} on {month} {day}, {year}, for one adult in economy. "
                    "Stop when matching flight options are visible.")
        else:
            back = min(28, day + rng.randint(2, 10))
            goal = (f"Find round-trip flights from {a} to {b}, departing {month} {day} and returning {month} {back}, {year}, "
                    "for one adult in economy. Stop when matching flight options are visible.")
        out.append({"url": "https://www.google.com/travel/flights?hl=en", "goal": goal, "source": "flights", "split": split})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnetnav", required=True, help="directory with nnetnav-v2 train.jsonl and test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flights", type=int, default=150, help="training flight tasks")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pairs = [(x, y) for x in CITIES for y in CITIES if x != y]
    rng.shuffle(pairs)
    topics = TOPICS[:]
    rng.shuffle(topics)
    tasks = nnetnav_tasks(f"{a.nnetnav}/train.jsonl", "train") + nnetnav_tasks(f"{a.nnetnav}/test.jsonl", "eval")
    tasks += flights(rng, pairs[: a.flights], "train") + flights(rng, pairs[a.flights: a.flights + 20], "eval")
    wiki = lambda t, s: {"url": "https://en.wikipedia.org/wiki/Main_Page", "source": "wikipedia", "split": s,
                         "goal": f"Find and open the Wikipedia article about {t}."}
    tasks += [wiki(t, "train") for t in topics[:30]] + [wiki(t, "eval") for t in topics[30:]]
    with open(a.out, "w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"id": f"task-{i:05d}", **t}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter((t["source"], t["split"]) for t in tasks))


if __name__ == "__main__":
    main()

"""Verified BibTeX via arXiv: search the exact title with the arXiv API, take arXiv's own BibTeX (complete author
list), and read the venue from the author-supplied journal_ref / comment when it names one."""
import re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

TITLES = {
 "deng2023mind2web": "Mind2Web: Towards a Generalist Agent for the Web",
 "murty2024nnetnav": "NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild",
 "zhou2024webarena": "WebArena: A Realistic Web Environment for Building Autonomous Agents",
 "yao2022webshop": "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents",
 "liu2018miniwob": "Reinforcement Learning on Web Interfaces Using Workflow-Guided Exploration",
 "zheng2024seeact": "GPT-4V(ision) is a Generalist Web Agent, if Grounded",
 "he2024webvoyager": "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models",
 "koh2024visualwebarena": "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks",
 "lu2024weblinx": "WebLINX: Real-World Website Navigation with Multi-Turn Dialogue",
 "yao2023react": "ReAct: Synergizing Reasoning and Acting in Language Models",
 "chen2023fireact": "FireAct: Toward Language Agent Fine-tuning",
 "zeng2024agenttuning": "AgentTuning: Enabling Generalized Agent Abilities for LLMs",
 "hsieh2023distilling": "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes",
 "hinton2015distilling": "Distilling the Knowledge in a Neural Network",
 "ross2011dagger": "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning",
 "guo2017calibration": "On Calibration of Modern Neural Networks",
 "kadavath2022know": "Language Models (Mostly) Know What They Know",
 "zhao2021calibrate": "Calibrate Before Use: Improving Few-Shot Performance of Language Models",
 "zheng2024selectors": "Large Language Models Are Not Robust Multiple Choice Selectors",
 "zheng2023judging": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
 "northcutt2021pervasive": "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks",
 "chen2023frugalgpt": "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance",
 "ong2025routellm": "RouteLLM: Learning to Route LLMs with Preference Data",
 "hu2022lora": "LoRA: Low-Rank Adaptation of Large Language Models",
 "warner2024modernbert": "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference",
 "yang2025qwen3": "Qwen3 Technical Report",
 "devlin2019bert": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
 "schick2023toolformer": "Toolformer: Language Models Can Teach Themselves to Use Tools",
 "gur2024realworld": "A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis",
 "cheng2026thisthat": "this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent",
 "wu2026reflex": "REFLEX with Jev for Efficient Selective Control in LLM Agents",
 "sun2026typesafe": "Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It",
}
VENUES = [("NeurIPS", "Advances in Neural Information Processing Systems"), ("Neural Information Processing Systems", "Advances in Neural Information Processing Systems"),
          ("ICLR", "International Conference on Learning Representations"), ("ICML", "International Conference on Machine Learning"),
          ("ACL", "Annual Meeting of the Association for Computational Linguistics"), ("EMNLP", "Conference on Empirical Methods in Natural Language Processing"),
          ("NAACL", "Conference of the North American Chapter of the Association for Computational Linguistics"), ("AISTATS", "International Conference on Artificial Intelligence and Statistics"),
          ("COLM", "Conference on Language Modeling")]
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wev-bib/0.1"}), timeout=60).read().decode()
        except Exception as e:
            err = e; time.sleep(5 + 5 * i)
    raise err

out, missing, report = [], [], []
for key, title in TITLES.items():
    q = " AND ".join(f"ti:{w}" for w in re.findall(r"[A-Za-z0-9]{4,}", title)[:8])
    feed = ET.fromstring(get("https://export.arxiv.org/api/query?max_results=10&search_query=" + urllib.parse.quote(q)))
    hit = None
    for e in feed.findall("a:entry", NS):
        t = " ".join(e.find("a:title", NS).text.split())
        if norm(t) == norm(title) or norm(t).startswith(norm(title)[:50]):
            hit = e; break
    if hit is None:
        missing.append(key); print("MISS", key, flush=True); time.sleep(3); continue
    aid = re.sub(r"v\d+$", "", hit.find("a:id", NS).text.rsplit("/abs/", 1)[1])
    note = " ".join(x.text or "" for x in hit.findall("x:journal_ref", NS) + hit.findall("x:comment", NS))
    bib = get(f"https://arxiv.org/bibtex/{aid}").strip()
    bib = re.sub(r"^@\w+\{[^,]+,", f"@misc{{{key},", bib, count=1)
    venue = next((full for short, full in VENUES if re.search(rf"\b{re.escape(short)}\b", note)), None)
    year = re.search(r"\b(20\d\d)\b", note)
    if venue:
        bib = bib.replace("@misc{", "@inproceedings{", 1)
        bib = re.sub(r"\n}\s*$", f",\n      booktitle={{{venue}}},\n      note={{arXiv:{aid}}}\n}}", bib)
        if year:
            bib = re.sub(r"year=\{\d+\}", f"year={{{year.group(1)}}}", bib)
    out.append(bib)
    report.append(f"{key:24s} {aid:12s} {'venue: ' + venue + ' ' + (year.group(1) if year else '') if venue else 'arXiv only'} | {note[:70]}")
    print(report[-1], flush=True)
    time.sleep(3)
open("refs_fetched.bib", "w").write("\n\n".join(out) + "\n")
print("MISSING:", missing)

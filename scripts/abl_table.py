"""Table of dev metrics for runs named <prefix><variant> (runs/*/metrics.json, logs/*.log)."""
import json, sys
prefix = sys.argv[1]
names = sys.argv[2:]
print(f"{'variant':<22} {'M2W step':>8} {'M2W click':>9} {'NN step':>8} {'NN op':>6} {'NN click':>8} {'NN DONE rec':>11} {'prem.DONE':>9} {'lat ms':>7} {'train r/s':>9}")
for v in names:
    h = json.load(open(f"runs/{prefix}{v}/metrics.json"))[-1]
    m, n = h["m2w-v2"], h["nnetnav-v2"]
    rs = [l for l in open(f"logs/{prefix}{v}.log") if "rec/s" in l][-1].split(" rec/s")[0].split()[-1]
    print(f"{v:<22} {m['step_success']:>8.3f} {m['click_target']['accuracy']:>9.3f} {n['step_success']:>8.3f} "
          f"{n['operation']['accuracy']:>6.3f} {n['click_target']['accuracy']:>8.3f} "
          f"{n['operation_by_gold']['DONE']['recall']:>11.3f} {n.get('premature_done_rate', 0):>9.3f} "
          f"{n['latency_ms_median']:>7.1f} {rs:>9}")

"""Quick empirical analysis of inferred-crash bias on a register campaign."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

run_dir = Path(sys.argv[1])
csv_path = run_dir / "results.csv"

rows_by_fid = defaultdict(list)
with csv_path.open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r.get("event") != "result":
            continue
        if r.get("phase") != "fault":
            continue
        fid = r.get("fault_id", "")
        if fid in ("", "-1", "-1.0"):
            continue
        try:
            img = int(float(r["image_id"]))
        except Exception:
            continue
        rows_by_fid[fid].append(
            (img, r["status"], r.get("inferred", "").lower() == "true")
        )

n_faults = len(rows_by_fid)
n_with_any_crash = 0
n_pure_clean = 0
n_first_crash_at = defaultdict(int)
real_ok_then_crash = 0
faults_w_real_ok_before_crash = 0

for fid, lst in rows_by_fid.items():
    lst.sort()
    statuses = [s for _, s, _ in lst]
    if any(s != "ok" for s in statuses):
        n_with_any_crash += 1
        first = next(
            (
                (i, s)
                for (i, s, inf) in lst
                if (not inf) and s != "ok"
            ),
            None,
        )
        if first is None:
            continue
        n_first_crash_at[first[0]] += 1
        # count real ok rows before the first real crash
        prior_oks = sum(
            1
            for (i, s, inf) in lst
            if (not inf) and i < first[0] and s == "ok"
        )
        if prior_oks > 0:
            faults_w_real_ok_before_crash += 1
            real_ok_then_crash += prior_oks
    else:
        n_pure_clean += 1

print(f"total faults     : {n_faults}")
print(f"faults all-ok    : {n_pure_clean}")
print(f"faults w/ crash  : {n_with_any_crash}")
print()
print("Distribution of FIRST real (non-inferred) crash image_id:")
buckets = [(0, 0), (1, 3), (4, 9), (10, 24), (25, 49), (50, 99)]
labels = ["img=0", "img 1-3", "img 4-9", "img 10-24", "img 25-49", "img 50-99"]
for (lo, hi), lab in zip(buckets, labels):
    c = sum(v for k, v in n_first_crash_at.items() if lo <= k <= hi)
    pct = 100.0 * c / max(1, n_with_any_crash)
    print(f"  {lab:10s}: {c:5d}  ({pct:5.1f}%)")
print()
print(f"Faults with at least one real OK before first real crash: "
      f"{faults_w_real_ok_before_crash}")
print(f"Total real OK rows that PRECEDED a real crash on the same fault: "
      f"{real_ok_then_crash}")
print()

late = sum(
    v * (99 - k) for k, v in n_first_crash_at.items() if k > 0
)
inf_total = sum(
    1 for lst in rows_by_fid.values() for _, _, inf in lst if inf
)
real_total = sum(
    1 for lst in rows_by_fid.values() for _, _, inf in lst if not inf
)
print(f"Real (dispatched) result rows : {real_total}")
print(f"Inferred (synthesized) rows   : {inf_total}")
print(
    f"Inferred rows where status is genuinely UNKNOWN (first crash at img>0): "
    f"{late}  ({100.0 * late / max(1, inf_total):.1f}% of all inferred)"
)

#!/usr/bin/env bash
# run_ablation.sh — train all variants across multiple seeds, report mean±std.
# Usage:  ./run_ablation.sh          (seeds 42 123 456)
#         ./run_ablation.sh 42       (single seed, fast check)
set -euo pipefail

PYTHON=".venv/bin/python"
WORKERS=4
SEEDS=(${@:-42 123 456})   # pass seeds as args, default to three seeds

log() { echo -e "\n\033[1;32m=== $* ===\033[0m\n"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree has uncommitted changes. Stash or commit before running."
fi

ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)

create_branch_if_missing() {
    local name=$1; shift
    if git show-ref --verify --quiet "refs/heads/$name"; then return; fi
    git checkout -b "$name" main
    for b in "$@"; do
        git merge --no-edit "$b" || die "Merge conflict in $name from $b. Resolve manually."
    done
    git checkout "$ORIG_BRANCH"
}

create_branch_if_missing exp/A+B   feature/A feature/B
create_branch_if_missing exp/A+B+C feature/A feature/B feature/C

run_variant() {
    local branch=$1 tag=$2 seed=$3
    log "Variant: $tag  seed=$seed  (branch: $branch)"
    git checkout "$branch"
    $PYTHON main.py --seed "$seed" --num_workers $WORKERS --exp_tag "${tag}_s${seed}"
    git checkout "$ORIG_BRANCH"
}

VARIANTS=(
    "main:baseline"
    "feature/A:A"
    "feature/B:B"
    "feature/C:C"
    "exp/A+B:A+B"
    "exp/A+B+C:A+B+C"
)

for seed in "${SEEDS[@]}"; do
    log "======  SEED $seed  ======"
    for entry in "${VARIANTS[@]}"; do
        branch="${entry%%:*}"
        tag="${entry##*:}"
        run_variant "$branch" "$tag" "$seed"
    done
done

# ── SVM & ResNet run once (deterministic enough) ─────────────────────────────
log "Baseline: SVM-cumulants (seed ${SEEDS[0]})"
$PYTHON baselines/svm_baseline.py --seed "${SEEDS[0]}" 2>&1 | tee training/svm_results.txt

log "Baseline: ResNet-18 1D (seed ${SEEDS[0]})"
$PYTHON baselines/train_resnet.py --seed "${SEEDS[0]}" --num_workers $WORKERS

# ── Collect results ───────────────────────────────────────────────────────────
log "Collecting results"
$PYTHON - << 'PYEOF'
import os, re, glob, json
from collections import defaultdict

results = defaultdict(list)
pattern = re.compile(r"overall accuracy is: ([\d.]+).*macro F1.*: ([\d.]+).*kappa.*: ([\d.]+)", re.DOTALL)

for log_file in sorted(glob.glob("training/*/log/log.txt")):
    tag = log_file.split("/")[1]          # e.g. 2016.10a_baseline_s42
    variant = re.sub(r"_s\d+$", "", tag)  # strip seed suffix
    variant = re.sub(r"^2016\.10a_", "", variant)
    with open(log_file) as f: content = f.read()
    m = pattern.search(content)
    if m:
        results[variant].append({
            "acc": float(m.group(1)),
            "f1":  float(m.group(2)),
            "kappa": float(m.group(3))
        })

print(f"\n{'Variant':<14} {'Acc mean±std':>16} {'F1 mean±std':>16} {'Kappa mean±std':>16}  n")
print("-" * 70)
for v in ["baseline","A","B","C","A+B","A+B+C"]:
    runs = results.get(v, [])
    if not runs: continue
    import statistics as st
    accs = [r["acc"] for r in runs]
    f1s  = [r["f1"]  for r in runs]
    kaps = [r["kappa"] for r in runs]
    def fmt(vals):
        m = sum(vals)/len(vals)
        s = st.stdev(vals) if len(vals)>1 else 0
        return f"{m*100:.2f}±{s*100:.2f}%"
    print(f"{v:<14} {fmt(accs):>16} {fmt(f1s):>16} {fmt(kaps):>16}  {len(runs)}")
PYEOF

log "All done. Results in training/"

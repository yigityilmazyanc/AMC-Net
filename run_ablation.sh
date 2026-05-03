#!/usr/bin/env bash
# run_ablation.sh — train all 6 AMC-Net variants sequentially then the two baselines.
# Run from repo root on a clean working tree:
#   chmod +x run_ablation.sh && ./run_ablation.sh
set -euo pipefail

PYTHON=".venv/bin/python"
SEED=42
WORKERS=4

log() { echo -e "\n\033[1;32m=== $* ===\033[0m\n"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# Require clean working tree so branch switches are safe
if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree has uncommitted changes. Stash or commit before running."
fi

ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# ── Create combined branches once ───────────────────────────────────────────
create_branch_if_missing() {
    local name=$1; shift          # remaining args: branches to merge in order
    if git show-ref --verify --quiet "refs/heads/$name"; then
        echo "Branch $name already exists — skipping creation."
        return
    fi
    git checkout -b "$name" main
    for b in "$@"; do
        git merge --no-edit "$b" || die "Merge conflict creating $name from $b. Resolve manually."
    done
    git checkout "$ORIG_BRANCH"
}

create_branch_if_missing exp/A+B   feature/A feature/B
create_branch_if_missing exp/A+B+C feature/A feature/B feature/C

# ── Train one AMC-Net variant ────────────────────────────────────────────────
run_variant() {
    local branch=$1
    local tag=$2
    log "Variant: $tag  (branch: $branch)"
    git checkout "$branch"
    $PYTHON main.py --seed $SEED --num_workers $WORKERS --exp_tag "$tag"
    git checkout "$ORIG_BRANCH"
}

# ── 6 AMC-Net variants ────────────────────────────────────────────────────────
run_variant main          "baseline"
run_variant feature/A     "A"
run_variant feature/B     "B"
run_variant feature/C     "C"
run_variant exp/A+B       "A+B"
run_variant exp/A+B+C     "A+B+C"

# ── SVM-cumulants baseline ────────────────────────────────────────────────────
log "Baseline: SVM-cumulants"
$PYTHON baselines/svm_baseline.py --seed $SEED 2>&1 | tee training/svm_results.txt

# ── ResNet-18 baseline ────────────────────────────────────────────────────────
log "Baseline: ResNet-18 1D"
$PYTHON baselines/train_resnet.py --seed $SEED --num_workers $WORKERS

log "All done. Results in training/"

#!/usr/bin/env bash
# End-to-end smoke test on a synthetic repository:
# toy data -> autoencoder -> DMP -> anchors -> sampling (plain / steered / composite / composed),
# then the same with one autoencoder per task.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
OUT=${1:-runs/smoke}
CFG=configs/smoke.yaml
rm -rf "$OUT"

$PY scripts/make_toy_repository.py --out "$OUT/data"
$PY scripts/train_autoencoder.py --config $CFG --repository "$OUT/data/repository.pt" --out "$OUT/autoencoder.pt"
$PY scripts/train_dmp.py --config $CFG --repository "$OUT/data/repository.pt" \
    --text-encodings "$OUT/data/text_encodings.pt" --autoencoder "$OUT/autoencoder.pt" --out "$OUT/dmp.pt"
$PY scripts/select_anchors.py --repository "$OUT/data/repository.pt" \
    --accuracies "$OUT/data/train_accuracies.csv" --out "$OUT/anchors.pt"

COMMON=(--dmp "$OUT/dmp.pt" --autoencoder "$OUT/autoencoder.pt" --text-encodings "$OUT/data/text_encodings.pt"
        --repository "$OUT/data/repository.pt" --steps 20 --n-samples 4)
$PY scripts/sample.py "${COMMON[@]}" --tasks task0 --out "$OUT/task0.pt"
$PY scripts/sample.py "${COMMON[@]}" --tasks task0 --anchors "$OUT/anchors.pt" --out "$OUT/task0_steered.pt"
$PY scripts/sample.py "${COMMON[@]}" --tasks task0 task1 --anchors "$OUT/anchors.pt" --out "$OUT/pair_steered.pt"
$PY scripts/sample.py "${COMMON[@]}" --tasks task0 task1 --compose --scale 3.0 3.0 --out "$OUT/pair_composed.pt"

# per-task autoencoders: one autoencoder per toy task, a DMP on their latents, steered pair sampling
PER_TASK=()
for T in task0 task1 task2 task3; do
    $PY scripts/train_autoencoder.py --config $CFG --repository "$OUT/data/repository.pt" --tasks $T --out "$OUT/ae_$T.pt"
    PER_TASK+=("$T=$OUT/ae_$T.pt")
done
$PY scripts/train_dmp.py --config $CFG --repository "$OUT/data/repository.pt" \
    --text-encodings "$OUT/data/text_encodings.pt" --autoencoder "${PER_TASK[@]}" --out "$OUT/dmp_per_task.pt"
$PY scripts/sample.py "${COMMON[@]}" --dmp "$OUT/dmp_per_task.pt" --autoencoder "${PER_TASK[@]}" \
    --tasks task0 task1 --anchors "$OUT/anchors.pt" --decode-task task1 --out "$OUT/pair_per_task.pt"

$PY tests/check_smoke_outputs.py "$OUT"
echo "smoke test passed"

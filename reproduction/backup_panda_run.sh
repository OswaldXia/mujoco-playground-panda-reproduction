#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-finetune}"
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "full" && "$RUN_KIND" != "finetune" && "$RUN_KIND" != "official" && "$RUN_KIND" != "robustness" ]]; then
  echo "Usage: $0 [smoke|full|finetune|official|robustness]"
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${PANDA_ARTIFACT_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-$RUN_KIND}"
BACKUP_DIR="${PANDA_BACKUP_DIR:-$HOME/panda-reproduction-archives}"

stage() {
  echo
  echo "[$1/4] $2"
}

item() {
  printf "  %-20s %s\n" "$1" "$2"
}

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Artifact directory not found: $ARTIFACT_DIR"
  exit 1
fi

stage 1 "Validate the run"
item "Mode" "$RUN_KIND"
item "Artifact root" "$ARTIFACT_DIR"

REQUIRED_FILES=(manifest.json console.log evaluation-summary.json)
for filename in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$ARTIFACT_DIR/$filename" ]]; then
    echo "Required artifact is missing: $ARTIFACT_DIR/$filename"
    exit 1
  fi
done

shopt -s nullglob
RUN_DIRS=("$ARTIFACT_DIR"/runs/PandaPickCubeCartesian-*)
if (( ${#RUN_DIRS[@]} == 0 )); then
  echo "No PandaPickCubeCartesian run found under $ARTIFACT_DIR/runs."
  exit 1
fi
LATEST_RUN="${RUN_DIRS[0]}"
for run_dir in "${RUN_DIRS[@]}"; do
  if [[ "$run_dir" -nt "$LATEST_RUN" ]]; then
    LATEST_RUN="$run_dir"
  fi
done

CHECKPOINTS=("$LATEST_RUN"/checkpoints/[0-9]*)
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No checkpoints found under $LATEST_RUN/checkpoints."
  exit 1
fi
LATEST_CHECKPOINT="${CHECKPOINTS[0]}"
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ "$(basename "$checkpoint")" > "$(basename "$LATEST_CHECKPOINT")" ]]; then
    LATEST_CHECKPOINT="$checkpoint"
  fi
done
if [[ ! -f "$LATEST_CHECKPOINT/ppo_network_config.json" && ! -f "$LATEST_CHECKPOINT/config.json" ]]; then
  echo "Latest checkpoint is incomplete: $LATEST_CHECKPOINT"
  exit 1
fi

item "Latest run" "$LATEST_RUN"
item "Latest checkpoint" "$LATEST_CHECKPOINT"
item "Source size" "$(du -sh "$ARTIFACT_DIR" | awk '{print $1}')"

stage 2 "Prepare a non-overwriting archive"
mkdir -p "$BACKUP_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_NAME="panda-vision-$RUN_KIND-$STAMP.tar.gz"
ARCHIVE_PATH="$BACKUP_DIR/$ARCHIVE_NAME"
PARTIAL_PATH="$BACKUP_DIR/.$ARCHIVE_NAME.partial"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

if [[ -e "$ARCHIVE_PATH" || -e "$CHECKSUM_PATH" || -e "$PARTIAL_PATH" ]]; then
  echo "Backup target already exists; nothing was overwritten:"
  echo "  $ARCHIVE_PATH"
  exit 1
fi

cleanup_partial() {
  if [[ -f "$PARTIAL_PATH" ]]; then
    rm -f "$PARTIAL_PATH"
  fi
}
trap cleanup_partial EXIT

item "Backup directory" "$BACKUP_DIR"
item "Archive" "$ARCHIVE_PATH"

stage 3 "Create and verify the archive"
tar -C "$(dirname "$ARTIFACT_DIR")" \
  -czf "$PARTIAL_PATH" \
  "$(basename "$ARTIFACT_DIR")"
tar -tzf "$PARTIAL_PATH" >/dev/null
mv "$PARTIAL_PATH" "$ARCHIVE_PATH"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$BACKUP_DIR"
    sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
  )
elif command -v shasum >/dev/null 2>&1; then
  (
    cd "$BACKUP_DIR"
    shasum -a 256 "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
  )
else
  echo "Neither sha256sum nor shasum is available; archive exists but checksum creation failed."
  exit 1
fi

stage 4 "Backup complete"
item "Archive" "$ARCHIVE_PATH"
item "Checksum" "$CHECKSUM_PATH"
item "Archive size" "$(du -sh "$ARCHIVE_PATH" | awk '{print $1}')"
echo
echo "Verify later with:"
if command -v sha256sum >/dev/null 2>&1; then
  echo "  cd '$BACKUP_DIR' && sha256sum -c '$ARCHIVE_NAME.sha256'"
else
  echo "  cd '$BACKUP_DIR' && shasum -a 256 -c '$ARCHIVE_NAME.sha256'"
fi

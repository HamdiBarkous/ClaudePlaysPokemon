#!/usr/bin/env bash
#
# Sync this project (the WSL working copy) to a runnable folder on the Windows
# Desktop, so the app can be run natively on Windows. WSL stays the place you
# edit and commit; run this whenever you want to push the latest code over.
#
# Mirrors the repo EXCEPT .venv — the Linux virtualenv is useless on Windows,
# and excluding it also protects a Windows-built .venv in the destination from
# being deleted by --delete (rsync never deletes excluded paths). So after the
# first sync you build the venv once on the Windows side and it survives every
# later sync.
#
# Usage:
#   scripts/sync_to_desktop.sh                 # sync to the default destination
#   scripts/sync_to_desktop.sh --dry-run       # preview without copying
#   scripts/sync_to_desktop.sh /mnt/c/path     # sync to a custom destination
#
set -euo pipefail

# Repo root = parent of this script's directory (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(dirname "$SCRIPT_DIR")"

DRYRUN=()
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
  DRYRUN=(--dry-run)
  shift
fi

DEST="${1:-/mnt/c/Users/hamdi/OneDrive/Desktop/pokemon}"

# Never copied. Add ".git/" here if you don't want a git repo on the Windows side.
EXCLUDES=(
  ".venv/"
  "__pycache__/"
  "*.pyc"
  ".pytest_cache/"
)

command -v rsync >/dev/null || {
  echo "rsync not found. Install it with: sudo apt install rsync" >&2
  exit 1
}

exclude_args=()
for e in "${EXCLUDES[@]}"; do exclude_args+=(--exclude="$e"); done

mkdir -p "$DEST"

echo "Syncing project -> Windows Desktop"
echo "  from:      $SRC/"
echo "  to:        $DEST/"
echo "  excluding: ${EXCLUDES[*]}"
echo

# -rlt (not -a): skip perm/owner/group preservation, which the Windows DrvFs
#   mount doesn't support and which otherwise spams errors.
# --delete: make the destination a mirror (excluded paths are kept).
# --modify-window=1: tolerate the 1s timestamp rounding on DrvFs so unchanged
#   files aren't needlessly re-copied.
rsync -rlt --delete --modify-window=1 -h -v "${DRYRUN[@]}" \
  "${exclude_args[@]}" "$SRC/" "$DEST/"

echo
if [[ ${#DRYRUN[@]} -gt 0 ]]; then
  echo "Dry run only — nothing was copied."
else
  echo "Done. On Windows (PowerShell), first time only:"
  echo "    cd \$env:USERPROFILE\\OneDrive\\Desktop\\pokemon"
  echo "    uv sync"
  echo "  then run:"
  echo "    uv run main.py --rom \"Pokemon Red.gb\" --steps 1000 --display"
fi

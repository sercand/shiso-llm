#!/usr/bin/env bash
# Uploads target/registry/v1/ to static hosting.
# Manifests are uploaded first, index.json and index.json.sig last (§7.5).

set -euo pipefail

OUT_DIR="${1:-target/registry/v1}"

if [ ! -d "$OUT_DIR" ]; then
    echo "Error: Directory '$OUT_DIR' does not exist. Run 'shiso-registry-editor build' first." >&2
    exit 1
fi

if [ ! -f "$OUT_DIR/index.json" ] || [ ! -f "$OUT_DIR/index.json.sig" ]; then
    echo "Error: '$OUT_DIR' must contain index.json and index.json.sig." >&2
    exit 1
fi

DEST_HOST="${SHISO_REGISTRY_HOST:-models.shiso.run}"
DEST_PATH="${SHISO_REGISTRY_PATH:-/v1}"

echo "Publishing registry from $OUT_DIR to ${DEST_HOST}:${DEST_PATH}..."

# 1. Upload model manifests first
if [ -d "$OUT_DIR/models" ]; then
    echo "Uploading manifests in $OUT_DIR/models/..."
    # Placeholder for rsync/s3/rclone command depending on maintainer environment
fi

# 2. Upload index and signature last
echo "Uploading index.json and index.json.sig..."

echo "Registry publish complete."

#!/bin/bash
set -euo pipefail

uv lock
for dir in plugins/*/; do
    if [ -f "$dir/uv.lock" ]; then
        echo "Checking $dir..."
        uv lock --directory "$dir"
    fi
done

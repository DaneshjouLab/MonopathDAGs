#!/bin/bash

SRC_DIR="src"
DEST_DIR="tests"
FORCE=false
TMP_TRACKED=$(mktemp)

# Handle --force flag
if [[ "$1" == "--force" ]]; then
    FORCE=true
    echo "⚠️  Overwriting ALL test files (forced mode)"
fi

# Step 1: Create empty test files
find "$SRC_DIR" -type f -name "*.py" ! -name "__init__.py" | while read -r src_file; do
    rel_path="${src_file#$SRC_DIR/}"
    src_dir=$(dirname "$rel_path")
    src_base=$(basename "$rel_path")

    # Skip hidden files or __pycache__
    [[ "$src_dir" == *"__pycache__"* || "$src_base" == .* ]] && continue

    test_dir="$DEST_DIR/$(echo "$src_dir" | sed 's|[^/]*|test_&|g')"
    test_file="$test_dir/test_${src_base}"

    echo "$test_file" >> "$TMP_TRACKED"
    mkdir -p "$test_dir"

    if [[ "$FORCE" == true || ! -s "$test_file" ]]; then
        > "$test_file"
        echo "➕ Created/Overwritten: $test_file"
    else
        echo "✔️  Skipped existing non-empty: $test_file"
    fi
done

# Step 2: Delete test files that no longer map to any src file
find "$DEST_DIR" -type f -name "test_*.py" | while read -r test_file; do
    if ! grep -Fxq "$test_file" "$TMP_TRACKED"; then
        echo "🗑️  Deleted obsolete: $test_file"
        rm -f "$test_file"
    fi
done

rm "$TMP_TRACKED"

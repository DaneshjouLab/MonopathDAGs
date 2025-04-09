#!/bin/bash

SRC_DIR="src"
DEST_DIR="tests"
FORCE=false
TMP_TRACKED=$(mktemp)

if [[ "$1" == "--force" ]]; then
    FORCE=true
    echo "⚠️  Forcing overwrite of all test files"
fi
generate_test_skeleton() {
    local src_file="$1"
    local test_file="$2"

    local in_class=0
    local class_name=""
    local indent=""

    echo "" > "$test_file"

    while IFS= read -r line; do
        # Match class definitions
        if [[ "$line" =~ ^([[:space:]]*)class[[:space:]]+([A-Za-z_][A-Za-z0-9_]*) ]]; then
            indent="${BASH_REMATCH[1]}"
            class_name="${BASH_REMATCH[2]}"
            echo -e "class Test${class_name}:" >> "$test_file"
            in_class=1
            continue
        fi

        # Match top-level function (only if not inside a class)
        if [[ "$in_class" -eq 0 && "$line" =~ ^[[:space:]]*def[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*) ]]; then
            func="${BASH_REMATCH[1]}"
            [[ "$func" == "__init__" ]] && continue
            echo -e "def test_${func}():\n    pass\n" >> "$test_file"
            continue
        fi

        # Match method inside a class
        if [[ "$in_class" -eq 1 && "$line" =~ ^${indent}[[:space:]]{4}def[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*) ]]; then
            method="${BASH_REMATCH[1]}"
            [[ "$method" == "__init__" ]] && continue
            echo -e "    def test_${method}(self):\n        pass\n" >> "$test_file"
            continue
        fi

        # Detect end of class (non-indented line after class)
        if [[ "$in_class" -eq 1 && ! "$line" =~ ^${indent}[[:space:]] ]]; then
            echo "" >> "$test_file"
            in_class=0
        fi
    done < "$src_file"
}


# Step 1: Create or update test skeletons
find "$SRC_DIR" -type f -name "*.py" ! -name "__init__.py" | while read -r src_file; do
    rel_path="${src_file#$SRC_DIR/}"
    src_dir=$(dirname "$rel_path")
    src_base=$(basename "$rel_path")

    [[ "$src_dir" == *"__pycache__"* || "$src_base" == .* ]] && continue

    test_dir="$DEST_DIR/$(echo "$src_dir" | sed 's|[^/]*|test_&|g')"
    test_file="$test_dir/test_${src_base}"

    echo "$test_file" >> "$TMP_TRACKED"
    mkdir -p "$test_dir"

    if [[ "$FORCE" == true || ! -s "$test_file" ]]; then
        generate_test_skeleton "$src_file" "$test_file"
        echo "🧪 Skeleton created: $test_file"
    else
        echo "✔️  Skipped existing non-empty: $test_file"
    fi
done

# Step 2: Delete obsolete files
find "$DEST_DIR" -type f -name "test_*.py" | while read -r test_file; do
    if ! grep -Fxq "$test_file" "$TMP_TRACKED"; then
        echo "🗑️  Deleted obsolete: $test_file"
        rm -f "$test_file"
    fi
done

rm "$TMP_TRACKED"

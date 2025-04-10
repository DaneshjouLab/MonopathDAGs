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

    python3 - "$src_file" "$test_file" <<EOF
import ast
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

def generate(src_path, test_path):
    with open(src_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            lines.append(f"def test_{node.name}():")
            lines.append("    pass\n")

        elif isinstance(node, ast.ClassDef):
            lines.append(f"class Test{node.name}:")

            methods = [
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name != "__init__"
            ]

            if methods:
                for method in methods:
                    lines.append(f"    def test_{method.name}(self):")
                    lines.append("        pass\n")
            else:
                lines.append("    def test_placeholder(self):")
                lines.append("        pass\n")

            lines.append("")  # extra newline

    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w") as f:
        f.write("\\n".join(lines))
        f.write("\\n")

generate(src, dst)
EOF
}

# Step 1: Walk src and build test files
find "$SRC_DIR" -type f -name "*.py" ! -name "__init__.py" | while read -r src_path; do
    rel_path="${src_path#$SRC_DIR/}"
    src_dir=$(dirname "$rel_path")
    src_base=$(basename "$rel_path")

    [[ "$src_dir" == *"__pycache__"* || "$src_base" == .* ]] && continue

    test_dir="$DEST_DIR/$(echo "$src_dir" | sed 's|[^/]*|test_&|g')"
    test_file="$test_dir/test_${src_base}"

    echo "$test_file" >> "$TMP_TRACKED"
    mkdir -p "$test_dir"

    if [[ "$FORCE" == true || ! -s "$test_file" ]]; then
        generate_test_skeleton "$src_path" "$test_file"
        echo "🧪 Generated: $test_file"
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

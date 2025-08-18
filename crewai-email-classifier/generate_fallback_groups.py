# generate_fallback_groups.py

import yaml
import os

def generate_classification_fallback_groups(
    input_path="config/classification_rules.yaml",
    output_path="config/classification_fallback_groups.yaml"
):
    try:
        # ✅ Load YAML classification rules
        with open(input_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        grouped = {}
        for entry in data.get("classifications", []):
            group = entry.get("group")
            name = entry.get("name")
            if group and name:
                grouped.setdefault(group, []).append(name)

        output = {"fallback_groups": grouped}

        # ✅ Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ✅ Save fallback groups to output YAML
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, sort_keys=False, allow_unicode=True)

        print(f"✅ classification_fallback_groups.yaml written to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate fallback groups: {e}")

# Run this only once to regenerate
if __name__ == "__main__":
    generate_classification_fallback_groups()

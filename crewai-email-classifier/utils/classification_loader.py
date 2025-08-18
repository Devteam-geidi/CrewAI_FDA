import yaml
import os

def load_yaml_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        print(f"❌ Error loading YAML file at {path}: {e}")
        return {}

def load_classifications(yaml_path="config/classification_rules.yaml") -> list:
    data = load_yaml_file(yaml_path)
    return data.get("classifications", [])

def load_known_routes(yaml_path="config/classification_rules.yaml") -> set:
    return {item.get("name") for item in load_classifications(yaml_path) if item.get("name")}

def load_fallback_groups(yaml_path="config/classification_fallback_groups.yaml") -> dict:
    data = load_yaml_file(yaml_path)
    return data.get("fallback_groups", {})

def load_fallback_groups_from_rules(yaml_path="config/classification_rules.yaml") -> dict:
    groups = {}
    for item in load_classifications(yaml_path):
        group = item.get("group")
        name = item.get("name")
        if group and name:
            groups.setdefault(group, []).append(name)
    return groups

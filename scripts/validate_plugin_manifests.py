#!/usr/bin/env python3
"""
Validate Claude Code plugin and marketplace manifests.

This script validates the JSON structure of plugin.json and marketplace.json files
according to the Claude Code schema requirements.

Usage:
    python scripts/validate_plugin_manifests.py
"""

import json
import sys
from pathlib import Path


def validate_plugin_json(plugin_json_path: Path) -> bool:
    """Validate plugin.json structure."""
    if not plugin_json_path.exists():
        print(f"Error: {plugin_json_path} not found")
        return False

    try:
        with open(plugin_json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {plugin_json_path}: {e}")
        return False

    required_fields = ["name", "version", "description", "schema_version"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print(f"Error: Missing required fields in plugin.json: {missing_fields}")
        return False

    print(f"✓ {plugin_json_path} is valid")
    return True


def validate_marketplace_json(marketplace_json_path: Path) -> bool:
    """Validate marketplace.json structure."""
    if not marketplace_json_path.exists():
        print(f"Info: {marketplace_json_path} does not exist, skipping validation")
        return True

    try:
        with open(marketplace_json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {marketplace_json_path}: {e}")
        return False

    # Check required root fields
    required_fields = ["name", "version", "description", "owner", "plugins"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print(f"Error: Missing required fields in marketplace.json: {missing_fields}")
        return False

    # Validate owner object
    if not isinstance(data["owner"], dict):
        print("Error: 'owner' must be an object")
        return False

    owner_required = ["name"]
    owner_missing = [field for field in owner_required if field not in data["owner"]]
    if owner_missing:
        print(f"Error: Missing required owner fields: {owner_missing}")
        return False

    # Validate plugins array
    if not isinstance(data["plugins"], list):
        print("Error: 'plugins' must be an array")
        return False

    if len(data["plugins"]) == 0:
        print("Warning: plugins array is empty")

    # Validate each plugin
    for i, plugin in enumerate(data["plugins"]):
        if not isinstance(plugin, dict):
            print(f"Error: Plugin {i} is not an object")
            return False

        plugin_required = ["name", "description", "source"]
        plugin_missing = [field for field in plugin_required if field not in plugin]
        if plugin_missing:
            print(f"Error: Plugin {i} missing required fields: {plugin_missing}")
            return False

        # Validate source field
        source = plugin["source"]
        if isinstance(source, str):
            # Relative path source
            pass
        elif isinstance(source, dict):
            # Object source (e.g., GitHub)
            if "source" not in source:
                print(f"Error: Plugin {i} source object missing 'source' field")
                return False
        else:
            print(f"Error: Plugin {i} source must be a string or object")
            return False

    print(f"✓ {marketplace_json_path} is valid")
    return True


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    plugin_json_path = project_root / ".claude-plugin" / "plugin.json"
    marketplace_json_path = project_root / ".claude-plugin" / "marketplace.json"

    plugin_valid = validate_plugin_json(plugin_json_path)
    marketplace_valid = validate_marketplace_json(marketplace_json_path)

    if not plugin_valid or not marketplace_valid:
        sys.exit(1)

    print("\nAll manifests are valid!")


if __name__ == "__main__":
    main()

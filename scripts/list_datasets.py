#!/usr/bin/env python3
"""List datasets in Roboflow workspace."""
import argparse
from roboflow import Roboflow

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", type=str, required=True)
args = parser.parse_args()

rf = Roboflow(api_key=args.api_key)
workspace = rf.workspace()

print(f"\n📦 Workspace: {workspace.name}")
print(f"📊 Available Projects:\n")

try:
    projects = workspace.projects()
    for project in projects.values():
        print(f"  - {project.name}")
        print(f"    ID: {project.project}")
except Exception as e:
    print(f"Error accessing projects: {e}")
    print("\nTry checking your workspace directly at:")
    print("https://app.roboflow.com/workspace")

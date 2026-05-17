#!/usr/bin/env python3
"""Check available versions for a Roboflow project."""
import argparse
from roboflow import Roboflow

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", type=str, required=True)
parser.add_argument("--project", type=str, required=True)
args = parser.parse_args()

try:
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace().project(args.project)

    print(f"Project: {project.name}")
    print(f"Project object: {project}")
    print(f"\nAvailable attributes:")
    print([attr for attr in dir(project) if not attr.startswith('_')])

    # Try to access versions
    try:
        print(f"\nVersions: {project.versions}")
    except Exception as e:
        print(f"Error accessing versions: {e}")

except Exception as e:
    print(f"Error: {e}")

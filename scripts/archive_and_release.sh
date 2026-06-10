#!/usr/bin/env bash
set -euo pipefail

# Ensure Docker image exists
if ! docker image inspect prediction-mania > /dev/null 2>&1; then
  echo "Docker image 'prediction-mania' not found. Please build it first." >&2
  exit 1
fi

# Archive the image
docker save prediction-mania -o prediction-mania.tar

echo "Docker image archived to prediction-mania.tar"

# Git operations
git add .
git commit -m "Add Docker image archive for release v1.0" || echo "No changes to commit."

git push origin main

# Tag and release
git tag -a v1.0 -m "Release v1.0"
git push origin v1.0

# Create GitHub release with asset (requires gh CLI)
if command -v gh > /dev/null; then
  gh release create v1.0 prediction-mania.tar --title "Prediction Mania v1.0" --notes "Docker image archive for Prediction Mania." --verify-tag
else
  echo "GitHub CLI (gh) not installed. Skipping release creation." >&2
fi

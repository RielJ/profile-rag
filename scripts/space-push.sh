#!/usr/bin/env bash
# Push the API image inputs to a Hugging Face Space (docs/DEPLOY.md).
#
#   HF_SPACE=<user>/<space> scripts/space-push.sh     (or: just space-push <user>/<space>)
#
# The Space gets the committed HEAD of Dockerfile, .dockerignore and backend/
# (no tests), plus a README with the front matter Spaces needs. Nothing else
# from this repository goes there: no .env, corpus/, eval/ or frontend/.
# Uncommitted changes are not pushed; commit first.
#
# Git needs push access to huggingface.co: run `hf auth login` once and let it
# set the git credential, or use a token with write access.
set -euo pipefail

: "${HF_SPACE:?set HF_SPACE to <user>/<space>, for example RielJ/profile-rag-api}"
# HF_SPACE_URL overrides the remote (a local bare repository, for testing).
remote="${HF_SPACE_URL:-https://huggingface.co/spaces/$HF_SPACE}"
repo_root="$(git rev-parse --show-toplevel)"
paths=(Dockerfile .dockerignore backend/pyproject.toml backend/uv.lock backend/README.md backend/src)

cd "$repo_root"
if ! git diff --quiet HEAD -- "${paths[@]}"; then
  echo "Warning: uncommitted changes under ${paths[*]} are not pushed." >&2
fi
commit="$(git rev-parse --short HEAD)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> Cloning the Space $HF_SPACE"
git clone --quiet --depth 1 "$remote" "$work/space"

# Replace the Space's files with this commit's. .git stays.
find "$work/space" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD "${paths[@]}" | tar -x -C "$work/space"

cat > "$work/space/README.md" <<EOF
---
title: profile-rag API
emoji: 🔎
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

The API of [profile-rag](https://github.com/RielJ/profile-rag), built from
commit $commit. The source and docs live in that repository; this Space only
holds what the Docker image needs.
EOF

cd "$work/space"
git add --all
if git diff --cached --quiet; then
  echo "==> The Space already has commit $commit. Nothing to push."
  exit 0
fi
git -c user.name="$(git -C "$repo_root" config user.name)" \
  -c user.email="$(git -C "$repo_root" config user.email)" \
  commit --quiet -m "Deploy profile-rag $commit"
echo "==> Pushing to $HF_SPACE"
git push --quiet origin HEAD
echo "==> Pushed. The Space rebuilds now: https://huggingface.co/spaces/$HF_SPACE"

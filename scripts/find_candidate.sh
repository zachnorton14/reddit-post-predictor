#!/usr/bin/env bash
# Print the id of the train-eval run whose candidate represents <commit>, or fail with an error.
#
#   scripts/find_candidate.sh <commit>
#
# That is the newest passing train-eval run from a push to `dev` whose commit is an ancestor of <commit>,
# provided no model file changed between the two. Commits that change only docs or workflows reuse the last
# trained candidate. Used by release-check.yml (required on `release`) and deploy.yml. Needs GH_TOKEN and a
# full git history (actions/checkout with fetch-depth: 0).
set -euo pipefail

head=${1:?usage: find_candidate.sh <commit>}
# Keep in sync with the push paths in .github/workflows/train-eval.yml.
MODEL_PATHS=(':(glob)*.py' 'pipeline/' 'docker/train.Dockerfile' 'requirements.txt')

runs=$(gh run list --workflow train-eval.yml --branch dev --event push --status success --limit 100 \
  --json databaseId,headSha --jq '.[] | "\(.databaseId) \(.headSha)"')

while read -r id sha; do
  [ -n "$id" ] || continue
  git cat-file -e "$sha^{commit}" 2>/dev/null || continue
  git merge-base --is-ancestor "$sha" "$head" || continue
  if git diff --quiet "$sha" "$head" -- "${MODEL_PATHS[@]}"; then
    echo "$id"
    exit 0
  fi
  echo "::error::Model files changed after the last trained commit ${sha:0:7} (run $id)." \
    "Wait for train-eval on dev to pass for the newest model change." >&2
  git diff --name-only "$sha" "$head" -- "${MODEL_PATHS[@]}" | sed 's/^/  changed: /' >&2
  exit 1
done <<< "$runs"

echo "::error::No passing train-eval run on dev for an ancestor of ${head:0:7}." >&2
exit 1

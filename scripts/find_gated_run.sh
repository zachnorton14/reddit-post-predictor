#!/usr/bin/env bash
# Print the id of the train-eval run whose model should be deployed for a commit on `release`.
#
#   scripts/find_gated_run.sh <commit>
#
# CI trains and gates every pull request into `dev`. `release` receives dev as a whole, so the model to deploy is
# the one from the newest pull request merged into dev whose merge commit is in <commit>'s history. That is only
# the newest code if every pull request was up to date with dev when it merged ("Require branches to be up to
# date before merging" on dev).
#
# Needs: gh (GH_TOKEN), and a git checkout with full history (actions/checkout fetch-depth: 0).
# Writes `id`, `pr` and `head` to $GITHUB_OUTPUT when set; prints the run id on stdout.
set -euo pipefail

commit="${1:?usage: find_gated_run.sh <commit>}"

# Newest merge first. 100 recent pull requests is plenty: release is promoted from dev often.
prs=$(gh pr list --base dev --state merged --limit 100 \
  --json number,headRefOid,mergeCommit,mergedAt \
  --jq 'sort_by(.mergedAt) | reverse | .[] | select(.mergeCommit != null) | "\(.number) \(.headRefOid) \(.mergeCommit.oid)"')

pr="" head=""
while read -r number sha merge; do
  [ -n "$number" ] || continue
  if git merge-base --is-ancestor "$merge" "$commit" 2>/dev/null; then
    pr="$number" head="$sha"
    break
  fi
done <<< "$prs"

if [ -z "$pr" ]; then
  echo "::error::no pull request merged into dev is in the history of $commit"
  exit 1
fi

id=$(gh run list --workflow train-eval.yml --commit "$head" --status success --limit 1 \
  --json databaseId --jq '.[0].databaseId // empty')
if [ -z "$id" ]; then
  echo "::error::no successful train-eval run for pull request #$pr (head $head)"
  exit 1
fi

echo "Deploying the model from pull request #$pr (head $head, train-eval run $id)" >&2
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  { echo "id=$id"; echo "pr=$pr"; echo "head=$head"; } >> "$GITHUB_OUTPUT"
fi
echo "$id"

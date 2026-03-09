#!/usr/bin/env bash
set -euo pipefail

# Monitor PRs 15 and 17 for CodeRabbit reviews and issues (no external deps like jq).
END=$(( $(date +%s) + 15 * 60 ))
PR15_ISSUES=1
PR17_ISSUES=1
PR15_NEW=0
PR17_NEW=0

while [ $(date +%s) -lt "$END" ]; do
  PR15=$(gh pr view 15 --json reviews,commits)
  PR17=$(gh pr view 17 --json reviews,commits)

  # PR 15 parsing using Python (no jq)
  PR15_JSON="$PR15"
  PR15_PARSE=$(python - << 'PY'
import json, sys
pr = json.loads(sys.argv[1])
commit_date = ''
if pr.get('commits') and len(pr['commits'])>0:
  commit_date = pr['commits'][0].get('committedDate','')
last_review_at = ''
last_review_author = ''
if pr.get('reviews') and len(pr['reviews'])>0:
  last = pr['reviews'][-1]
  last_review_at = last.get('submittedAt','')
  last_review_author = last.get('author',{}).get('login','')
issues = 0
for r in pr.get('reviews',[]):
  if r.get('author',{}).get('login','') == 'coderabbitai':
    body = (r.get('body') or '').lower()
    if 'critical' in body or 'major' in body:
      issues += 1
new_review_after_commit = 1 if (last_review_at and commit_date and last_review_at > commit_date and last_review_author == 'coderabbitai') else 0
print(f"{commit_date}|{last_review_at}|{last_review_author}|{new_review_after_commit}|{issues}")
PY
)
  PR15_COMMIT_DATE=$(echo "$PR15_PARSE" | cut -d'|' -f1)
  PR15_LAST_REVIEW_AT=$(echo "$PR15_PARSE" | cut -d'|' -f2)
  PR15_LAST_REVIEW_AUTHOR=$(echo "$PR15_PARSE" | cut -d'|' -f3)
  PR15_NEW_REVIEW=$(echo "$PR15_PARSE" | cut -d'|' -f4)
  PR15_ISSUES=$(echo "$PR15_PARSE" | cut -d'|' -f5)

  # PR 17 parsing using Python (no jq)
  PR17_JSON="$PR17"
  PR17_PARSE=$(python - << 'PY'
import json, sys
pr = json.loads(sys.argv[1])
commit_date = ''
if pr.get('commits') and len(pr['commits'])>0:
  commit_date = pr['commits'][0].get('committedDate','')
last_review_at = ''
last_review_author = ''
if pr.get('reviews') and len(pr['reviews'])>0:
  last = pr['reviews'][-1]
  last_review_at = last.get('submittedAt','')
  last_review_author = last.get('author',{}).get('login','')
issues = 0
for r in pr.get('reviews',[]):
  if r.get('author',{}).get('login','') == 'coderabbitai':
    body = (r.get('body') or '').lower()
    if 'critical' in body or 'major' in body:
      issues += 1
new_review_after_commit = 1 if (last_review_at and commit_date and last_review_at > commit_date and last_review_author == 'coderabbitai') else 0
print(f"{commit_date}|{last_review_at}|{last_review_author}|{new_review_after_commit}|{issues}")
PY
)
  PR17_COMMIT_DATE=$(echo "$PR17_PARSE" | cut -d'|' -f1)
  PR17_LAST_REVIEW_AT=$(echo "$PR17_PARSE" | cut -d'|' -f2)
  PR17_LAST_REVIEW_AUTHOR=$(echo "$PR17_PARSE" | cut -d'|' -f3)
  PR17_NEW_REVIEW=$(echo "$PR17_PARSE" | cut -d'|' -f4)
  PR17_ISSUES=$(echo "$PR17_PARSE" | cut -d'|' -f5)

  echo "PR 15: latest_commit=$PR15_COMMIT_DATE last_review_at=$PR15_LAST_REVIEW_AT new_review_from_CodeRabbit=$PR15_NEW_REVIEW issues=$PR15_ISSUES"
  echo "PR 17: latest_commit=$PR17_COMMIT_DATE last_review_at=$PR17_LAST_REVIEW_AT new_review_from_CodeRabbit=$PR17_NEW_REVIEW issues=$PR17_ISSUES"

  if [ "$PR15_ISSUES" -eq 0 ] && [ "$PR17_ISSUES" -eq 0 ]; then
    echo "No critical/major CodeRabbit issues on both PRs. Exiting."
    exit 0
  fi
  if [ "$PR15_NEW_REVIEW" -eq 1 ]; then
    echo "CodeRabbit has reviewed PR 15 after latest commit."
  fi
  if [ "$PR17_NEW_REVIEW" -eq 1 ]; then
    echo "CodeRabbit has reviewed PR 17 (after rate limit)."
  fi

  sleep 120
done
echo "15-minute timeout reached. Final status inputs below:"
PR15_FINAL=$(gh pr view 15 --json reviews,comments,commits)
PR17_FINAL=$(gh pr view 17 --json reviews,comments,commits)
echo "$PR15_FINAL" | python -m json.tool
echo "$PR17_FINAL" | python -m json.tool

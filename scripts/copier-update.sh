#!/usr/bin/env bash
# Update against template changes, and commit the update if there are no conflicts.
# Arguments are passed on to `copier update` (e.g. --vcs-ref=HEAD).
set -euo pipefail

copier update --skip-answered "$@"

VERSION=$(sed -n 's/^_commit: *//p' .copier-answers.yml)
MSG="Update against template $VERSION"

# Copier marks files with conflicts as unmerged
CONFLICTS=$(git diff --name-only --diff-filter=U)
if [[ -n "$CONFLICTS" ]]; then
  echo
  echo "The update has conflicts in the following files:"
  echo "$CONFLICTS" | sed 's/^/  /'
  echo
  echo "Resolve them with 'git mergetool' (or edit the <<<<<<< / >>>>>>> markers by hand),"
  echo "then check that no markers are left and commit the update with:"
  echo "  ./scripts/conflicts.sh"
  echo "  git add --all"
  echo "  git commit -m \"$MSG\""
  if [[ "$(git config merge.tool || true)" == "vimdiff" ]]; then
    echo
    echo "vimdiff cheat sheet:"
    echo "  Layout: LOCAL (your project) | BASE (old template) | REMOTE (new template)"
    echo "          MERGED (the file to edit) below"
    echo "  ]c / [c          next / previous change"
    echo "  :diffget LO      take the hunk from LOCAL (or RE / BA for REMOTE / BASE),"
    echo "                   in the MERGED window"
    echo "  Ctrl-w w         next window"
    echo "  :wqa             save and finish this file"
    echo "  :cq              abort, leaving this file unresolved"
  fi
  exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Already up to date."
  exit 0
fi

git add --all
if ! git commit -m "$MSG"; then
  echo
  echo "The commit failed (see above). The update is staged; after fixing the problem, run:"
  echo "  git add --all"
  echo "  git commit -m \"$MSG\""
  exit 1
fi

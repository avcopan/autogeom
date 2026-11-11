#!/usr/bin/env bash

# Set up direnv, if present
if command -v direnv >/dev/null 2>&1; then
  direnv allow
else
  echo "Warning: direnv is not installed."
  echo "You can install direnv and then run: direnv allow"
fi

# Initialize git repository, unless this is already one (e.g. a clone of an existing project)
if [[ -e .git ]]; then
  NEW_REPO=false
  echo "Already a Git repository: skipping git init and the initial commit."
else
  NEW_REPO=true
  git init -b main
  git add --all
  if git config --get user.name >/dev/null 2>&1 && git config --get user.email >/dev/null 2>&1; then
    git commit -m "Initial commit"
  else
    echo "Warning: Git author identity is not configured."
    echo "You can configure it as follows:"
    echo '  git config --global user.name "Your Name"'
    echo '  git config --global user.email "you@example.com"'
    echo "Afterwards, you can make the initial commit with:"
    echo '  git commit -m "Initial commit"'
  fi
fi

# Install git hooks with lefthook
lefthook install

# Install all pixi environments
pixi install --all

# Offer to set Pixi's pinning strategy, so that `pixi add` adds lower bounds only
if ! pixi config list pinning-strategy --json | grep -q '"latest-up"'; then
  PIN_CMD="pixi config set pinning-strategy latest-up --global"
  if [[ -t 0 ]]; then
    read -r -p "Set Pixi's pinning strategy to \"latest-up\" for all your projects? [Y/n] " answer
  else
    answer=n
  fi
  if [[ "${answer:-y}" =~ ^[Yy] ]]; then
    $PIN_CMD
  else
    echo "To set Pixi's pinning strategy later, run: $PIN_CMD"
  fi
  echo
fi

# Offer to use vimdiff as Git's merge tool, e.g. for conflicts from `pixi run copier-update`
if [[ -z "$(git config --global merge.tool || true)" ]]; then
  MERGE_CMD="git config --global merge.tool vimdiff"
  if [[ -t 0 ]]; then
    read -r -p "Set Git's merge tool to vimdiff for all your projects? [Y/n] " answer
  else
    answer=n
  fi
  if [[ "${answer:-y}" =~ ^[Yy] ]]; then
    $MERGE_CMD
  else
    echo "To set Git's merge tool later, run: $MERGE_CMD"
  fi
  echo
fi

echo "Setup complete."
if $NEW_REPO; then
  echo
  echo "Now, run the following commands to set up your remote repository:"
  echo "  git remote add origin git@github.com:avcopan/autogeom.git"
  echo "  git push -u origin main"
  echo
  echo "Then, link your documentation site under 'About' on your GitHub repository site"
  echo "(see https://github.com/avcopan/copier-pixi-python#initialize-local-repository)."
fi

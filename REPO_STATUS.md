# Repository status: local checkout mismatch

You are right that `app.py` exists in your GitHub screenshot. In this local environment, however, those uploaded files are not present in the checked-out branch.

## What is present locally

Tracked files in `HEAD`:

- `.gitkeep`
- `REPO_STATUS.md`

No `app.py`, `index.html`, `rag.py`, or `tinyllama-career-counselor/` directory exists in this local checkout.

## Why this happens

Most likely causes:

1. The uploaded files were pushed to a different branch than this local branch (`work`).
2. The local clone has no configured remote, so it cannot fetch the newer commits.
3. The working copy in this environment is not the same repo instance as your screenshot.

## Verification commands run here

- `find . -maxdepth 3 -type f | sort`
- `git branch -a`
- `git log --oneline --decorate --graph --all -n 20`
- `git ls-tree --name-only -r HEAD`

## How to sync this checkout so 404 can be debugged

If your GitHub repo URL is available, run:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git fetch origin
git checkout <branch-containing-app-py>
git pull --ff-only
```

Then verify:

```bash
ls app.py index.html
```

At that point, run:

```bash
python app.py
```

and share the startup log + requested URL that returns 404, so route/static-path mismatch can be fixed directly.

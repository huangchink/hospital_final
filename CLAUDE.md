# CLAUDE.md — ntuh-eyetracking

Guidance for anyone working in this repository — **human contributors and AI coding agents alike**
(e.g. Claude Code used by the NTUH team and by BU). Read this before making changes.

This repo is the **NTUH Eye-Tracking Suite**: three Windows apps (`VA_center_opt`, `calibration`,
`replayer`) for VA/VF screening with Ganzin Sol glasses + webcam. Setup, how to run, and how to build
are in [`README.md`](README.md). This file is the **rules of engagement**.

---

## Development workflow (mandatory)

We use a **feature-branch → pull-request → review → manual-merge** model. There are two long-lived
branches:

- **`main`** — released / stable. Only updated by a release merge. Never commit here directly.
- **`develop`** — integration branch. All work merges here **via reviewed pull requests**, never by a
  direct push.

### Steps for every change

1. **Branch off `develop`:**
   ```
   git switch develop && git pull
   git switch -c feature/<short-topic>      # or fix/<...>, chore/<...>, docs/<...>
   ```
2. **Commit** small, focused changes with clear messages (see [Commits](#commit-conventions)).
3. **Push the branch** and **open a pull request into `develop`** on GitHub
   (`gh pr create --base develop`). Write what changed and why, and how it was verified.
4. **A human reviews the PR on GitHub and merges it by hand** after approval.

### Rules for AI agents (Claude bots) — non-negotiable

- **Never push to `develop` or `main`.** Work only on a `feature/…`, `fix/…`, `chore/…`, or
  `docs/…` branch.
- **Open the PR and stop.** Do **not** merge it — not `gh pr merge`, not the GitHub UI, not a
  fast-forward push. A human reviews and merges.
- **Do not force-push** shared branches or rewrite published history.
- If asked to "merge", interpret it as "open the PR for review" unless a human explicitly confirms
  they are performing/authorising the merge themselves.
- Keep PRs **small and single-purpose** so they are reviewable.

### Recommended best practice (repo settings)

- Enable **branch protection** on `main` and `develop`: require a PR, require ≥1 approving review,
  disallow direct pushes and force-pushes. This makes the rules above enforced, not just documented.
- Prefer **squash-merge** for feature PRs to keep `develop` history linear and readable.
- Delete the feature branch after merge (`gh pr merge --delete-branch`, run **by the human** merger).

## Commit conventions

- **Do NOT add a `Co-Authored-By: Claude …` trailer** to commit messages.
- Present-tense, imperative subject line (≤ ~72 chars); body explains the *why* and how it was
  verified. Reference the app when relevant (e.g. `VA_center_opt: …`).

## Code constraints (repo-specific)

- **Do not edit the vendored `gazefollower/` internals** — it is a vendored upstream library. Wrap or
  extend it from `ntuh/` instead.
- **Sol scene decode runs in an isolated child process on purpose.** The Ganzin SDK's native H.264
  decode can hard-crash; process isolation (`ntuh/sol/sol_child.py` + `preview_client.py`) contains it
  and the parent respawns the child. Do **not** move Sol decoding into the parent, and avoid adding
  per-frame load to the child (it degrades the crash-prone decode). A live-camera tester view was
  tried in VA v1.0.2 and reverted for exactly this reason.
- The three apps stay **import-light at module top level**: heavy imports (pygame/tk/mediapipe/
  gazefollower/SDK) live inside `main()`/functions because `multiprocessing 'spawn'` re-imports the
  entry module in the child.
- Every version-visible change ships with a **version bump** (see below).

## Terminology

- **"Sol glasses"** = Ganzin Sol smart eye-tracking glasses. **"Chronus"** = the paired phone app.
- Use **"gaze estimation"** (not "eye tracking") for the webcam screen-coordinate prediction task.
- **"Simplified / offset calibration"** = the Sol offset-calibration in this repo (1/3/5-pt, screen-
  space or camera-space).

## Developer tooling

Optional but recommended for contributors working with AI coding agents. Neither is needed to build
or run the apps.

### RTK — token-optimizing CLI proxy (https://github.com/rtk-ai/rtk)

A single Rust binary that filters/compresses command output before an agent reads it (big token
savings). Install one way, then wire the Claude Code hook:

```
# Windows: download the prebuilt binary from the GitHub Releases page and put it on PATH,
#          or:  cargo install --git https://github.com/rtk-ai/rtk
# macOS/Linux:  brew install rtk   (or the install.sh from the repo)

rtk init -g            # install the Claude Code auto-rewrite hook (then restart Claude Code)
rtk --version          # verify install
rtk init --show        # verify the hook
rtk gain               # token-savings dashboard
```

Config lives at `~/.config/rtk/config.toml`. If `rtk gain` errors, you may have a name-collision with
a different `rtk` on PATH — check `which rtk` / `where rtk`.

### Ponytail — code-minimization plugin (https://github.com/DietrichGebert/ponytail)

A Claude Code plugin that steers agents toward minimal, necessary code (YAGNI / "do we need this?").

```
/plugin marketplace add DietrichGebert/ponytail
/plugin install ponytail@ponytail
```

Usage: `/ponytail [lite|full|ultra|off]` to set intensity, `/ponytail-review` to review a diff for
over-engineering. Default mode via `PONYTAIL_DEFAULT_MODE` or `~/.config/ponytail/config.json`.

## Versioning & release flow

Each app has an **independent** version in `ntuh/version.py` → `APP_VERSIONS`, shown in its window
title. Scheme is `MAJOR.MINOR.PATCH`:

- **PATCH** — bug fix / no user-visible behavior change (a fix that *fills a gap* in an existing
  feature is a PATCH, even if it adds a window/UI).
- **MINOR** — new user-facing feature or option.
- **MAJOR** — breaking change to the workflow or data layout.

### Releasing (scripted local release)

The build needs Windows + the native stack + the vendored Ganzin wheel + produces ~2 GB, so releases
are built **locally** with `release.py` (not in CI).

1. On a `feature/…` (or `release/…`) branch, for each **changed** app:
   - bump its entry in `ntuh/version.py` and add a one-line **changelog** note in that file;
   - add a dated release note `doc/YYYYMMDD_release_note.txt` and list it at the top of
     `MANUAL_DOCS` in `stage_release.py`;
   - update the app's `doc/*.md` if behavior changed.
2. Open a PR into `develop`; review; a human merges.
3. Build + package from `develop`:
   ```
   python release.py --tag
   ```
   This cleans/builds all three apps, runs `stage_release.py`, writes
   `release/YYYYMMDD_NTUH_EyeTracking_Suite.zip` (date-first, git-ignored), and creates per-app git
   tags `VA_center_opt-vX.Y.Z`, `calibration-vX.Y.Z`, `replayer-vX.Y.Z` for the current versions.
4. **A human** pushes the tags (`git push --tags`) and, for a stable release, opens a PR to merge
   `develop → main`. Hand off the zip from `release/`.

### Release artifact conventions

- Release zips live in the **git-ignored `release/`** folder and are named **date-first**:
  `YYYYMMDD_NTUH_EyeTracking_Suite.zip`. Keep only the **current** build there; remove superseded
  archives.
- Git tags are **per app**: `<app>-vX.Y.Z` (e.g. `VA_center_opt-v1.0.1`).

## Where things live

See [`README.md`](README.md#repository-layout). Quick pointers: app code in `ntuh/`; versions in
`ntuh/version.py`; build in `*.spec` + `build_exe.bat` + `stage_release.py`; release in `release.py`;
end-user docs in `doc/`.

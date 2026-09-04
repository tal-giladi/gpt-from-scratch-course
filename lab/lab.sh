#!/usr/bin/env bash
# The grader. Every lesson's exercise lives in lab/exercises/lesson_NN.py and is checked by
# lab/checks/NN.py against the REAL autoresearch code mounted at /autoresearch — so passing
# means your code agrees with the repo under study, not merely that it runs.
#
# Everything happens inside the autoresearch-cpu docker image. There is no venv and nothing
# is installed on the host.
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"

COMPOSE=(docker compose -f "$LAB_DIR/docker-compose.yml")
# Where the repo under study lives. By default it is the sibling clone two levels up, and
# docker-compose.yml finds it by relative path — which is what keeps this working on
# Windows, where an absolute Git Bash path (/c/Users/...) is not a path docker understands.
# Override only if you cloned it somewhere else, and then use a Windows-style path:
#   AUTORESEARCH_DIR='C:\code\autoresearch' bash lab/lab.sh check 01
AR_DIR="${AUTORESEARCH_DIR:-$LAB_DIR/../../../autoresearch}"

require_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop and try again." >&2
    exit 1
  fi
}

require_repo() {
  if [ ! -f "$AR_DIR/train.py" ]; then
    echo "Cannot find the autoresearch repo at: $AR_DIR" >&2
    echo "Clone it next to this course, or set AUTORESEARCH_DIR to where it lives." >&2
    exit 1
  fi
}

run_in_lab() {
  "${COMPOSE[@]}" run --rm --quiet-pull lab "$@"
}

cmd_up() {
  require_docker
  require_repo
  if ! docker image inspect autoresearch-cpu:latest >/dev/null 2>&1; then
    echo "Building the autoresearch CPU image (once) ..."
    docker compose -f "$AR_DIR/docker-compose.yml" build
  fi
  if ! docker volume inspect autoresearch_ar-cache >/dev/null 2>&1; then
    echo "Downloading data shards and training the tokenizer (once, ~185MB) ..."
    docker compose -f "$AR_DIR/docker-compose.yml" \
      run --rm autoresearch python prepare.py --num-shards 2
  fi
  echo "Verifying the lab can see the repo, the tokenizer and the data ..."
  run_in_lab python checks/00_selftest.py
  echo "Ready. Start with lessons/module-01/lesson-01.md, then: bash lab/lab.sh check 01"
}

cmd_check() {
  require_docker
  local nn="$1"
  [ -f "checks/${nn}.py" ] || { echo "No such check: checks/${nn}.py" >&2; exit 1; }
  run_in_lab python "checks/${nn}.py"
}

cmd_hint() {
  local nn="$1"
  echo "Open the lesson for exercise ${nn} (see _sidebar.md) and read its '## Hints' section."
  echo "The stub is lab/exercises/lesson_${nn}.py — every TODO says what is missing."
}

cmd_solve() {
  local nn="$1"
  cp "solutions/lesson_${nn}.py" "exercises/lesson_${nn}.py"
  echo "Copied the reference solution into exercises/lesson_${nn}.py."
  echo "Re-run: bash lab/lab.sh check ${nn}"
}

cmd_reset() {
  local nn="$1"
  git checkout -- "exercises/lesson_${nn}.py" 2>/dev/null ||
    echo "Could not git-restore exercises/lesson_${nn}.py — restore it from the repo manually."
}

cmd_status() {
  require_docker
  run_in_lab python checks/_status.py
}

cmd_shell() {
  require_docker
  run_in_lab python
}

case "${1:-}" in
  up)     cmd_up ;;
  check)  cmd_check "${2:?usage: lab.sh check NN}" ;;
  hint)   cmd_hint "${2:?usage: lab.sh hint NN}" ;;
  solve)  cmd_solve "${2:?usage: lab.sh solve NN}" ;;
  reset)  cmd_reset "${2:?usage: lab.sh reset NN}" ;;
  status) cmd_status ;;
  shell)  cmd_shell ;;
  *)
    echo "Usage: bash lab/lab.sh {up|check NN|hint NN|solve NN|reset NN|status|shell}"
    exit 1
    ;;
esac

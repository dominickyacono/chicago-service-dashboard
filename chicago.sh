#!/usr/bin/env bash
# ============================================================
#  chicago.sh — Chicago Service Dashboard project runner
#  Usage: bash ~/Desktop/chicago.sh
# ============================================================

PROJECT_DIR="$HOME/Desktop/ChicagoProject"
VENV="$PROJECT_DIR/2026env"
PYTHON="$VENV/bin/python"
BOKEH="$VENV/bin/bokeh"

# ── Colours ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Helpers ──────────────────────────────────────────────────
ok()   { echo -e "${GREEN}✔  $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠  $*${RESET}"; }
err()  { echo -e "${RED}✖  $*${RESET}"; }
hr()   { echo -e "${CYAN}────────────────────────────────────────${RESET}"; }

check_deps() {
  if [ ! -d "$VENV" ]; then
    err "Virtual environment not found at $VENV"
    exit 1
  fi
  if [ ! -d "$PROJECT_DIR/.git" ]; then
    warn "No git repo detected — git options will be skipped."
  fi
}

run_pipeline() {
  hr
  echo -e "${BOLD}▶  Running pipeline.py …${RESET}"
  cd "$PROJECT_DIR" && "$PYTHON" pipeline.py
  local code=$?
  [ $code -eq 0 ] && ok "pipeline.py finished successfully." \
                  || err "pipeline.py exited with code $code."
  return $code
}

run_app() {
  hr
  echo -e "${BOLD}▶  Starting Bokeh dashboard (app.py) …${RESET}"
  echo -e "   ${CYAN}→ Open http://localhost:5006/app in your browser${RESET}"
  echo -e "   Press ${BOLD}Ctrl+C${RESET} to stop the server.\n"
  cd "$PROJECT_DIR" && "$BOKEH" serve app.py --show
}

git_pull() {
  hr
  echo -e "${BOLD}▶  Pulling latest from origin …${RESET}"
  cd "$PROJECT_DIR" && git pull origin "$(git rev-parse --abbrev-ref HEAD)"
  [ $? -eq 0 ] && ok "Pull complete." || err "Pull failed — check your connection or conflicts."
}

git_push() {
  hr
  echo -e "${BOLD}▶  Staging & pushing changes …${RESET}"
  cd "$PROJECT_DIR"

  # Show what's changed
  git status --short
  echo ""

  # Prompt for commit message
  printf "${YELLOW}Commit message (leave blank to cancel): ${RESET}"
  read -r msg
  if [ -z "$msg" ]; then
    warn "Commit message empty — push cancelled."
    return 1
  fi

  git add -A
  git commit -m "$msg"
  git push origin "$(git rev-parse --abbrev-ref HEAD)"
  [ $? -eq 0 ] && ok "Pushed to origin successfully." || err "Push failed."
}

full_workflow() {
  hr
  echo -e "${BOLD}🔄  Full workflow: pull → pipeline → push → launch app${RESET}"
  hr

  git_pull   || { err "Aborting workflow after pull failure."; return 1; }
  run_pipeline || { err "Aborting workflow after pipeline failure."; return 1; }
  git_push

  echo ""
  printf "${YELLOW}Launch the dashboard now? [y/N]: ${RESET}"
  read -r launch
  [[ "$launch" =~ ^[Yy]$ ]] && run_app
}

# ── Menu ─────────────────────────────────────────────────────
show_menu() {
  clear
  echo -e "${BOLD}${CYAN}"
  echo "  ╔══════════════════════════════════════╗"
  echo "  ║   🏙  Chicago Service Dashboard       ║"
  echo "  ╚══════════════════════════════════════╝"
  echo -e "${RESET}"
  echo -e "  ${BOLD}1)${RESET}  Run pipeline.py        ${CYAN}(fetch & process new data)${RESET}"
  echo -e "  ${BOLD}2)${RESET}  Launch app.py           ${CYAN}(start Bokeh dashboard)${RESET}"
  echo -e "  ${BOLD}3)${RESET}  Git pull                ${CYAN}(pull latest from GitHub)${RESET}"
  echo -e "  ${BOLD}4)${RESET}  Git push                ${CYAN}(commit & push changes)${RESET}"
  echo -e "  ${BOLD}5)${RESET}  Full workflow           ${CYAN}(pull → pipeline → push → app)${RESET}"
  echo -e "  ${BOLD}q)${RESET}  Quit"
  hr
  printf "  ${BOLD}Choose an option: ${RESET}"
}

# ── Main ─────────────────────────────────────────────────────
check_deps

while true; do
  show_menu
  read -r choice
  case "$choice" in
    1) run_pipeline ;;
    2) run_app ;;
    3) git_pull ;;
    4) git_push ;;
    5) full_workflow ;;
    q|Q) echo -e "\n${GREEN}Goodbye!${RESET}\n"; exit 0 ;;
    *) warn "Invalid option — try 1–5 or q." ;;
  esac
  echo ""
  printf "  ${YELLOW}Press Enter to return to menu …${RESET}"
  read -r
done

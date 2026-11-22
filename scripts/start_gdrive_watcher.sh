#!/usr/bin/env bash
# Manage the always-on Google Drive checkpoint auto-uploader daemon.
# Usage:
#   bash scripts/start_gdrive_watcher.sh start [--all-existing]
#   bash scripts/start_gdrive_watcher.sh stop
#   bash scripts/start_gdrive_watcher.sh status
#   bash scripts/start_gdrive_watcher.sh logs

set -u
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WATCHER_PY="${SCRIPT_DIR}/gdrive_checkpoint_watcher.py"
WATCH_DIR="${WATCH_DIR:-${PROJECT_ROOT}/checkpoints}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive:Video-LLaVA/checkpoints}"
PID_FILE="$HOME/.cache/videollava_gdrive_watcher.pid"
LOG_FILE="$HOME/gdrive_upload_watcher.log"

mkdir -p "$HOME/.cache"

ACTION="${1:-status}"
MAX_EXISTING="${MAX_EXISTING:-1}"
if [ "${2:-}" = "--all-existing" ]; then
    MAX_EXISTING=0
fi

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

case "$ACTION" in
    start)
        if is_running; then
            echo "GDrive watcher is already running (PID $(cat "$PID_FILE"))."
            exit 0
        fi
        PYTHON_BIN="python3"
        if [ -x "$HOME/miniconda3/envs/videollava/bin/python" ]; then
            PYTHON_BIN="$HOME/miniconda3/envs/videollava/bin/python"
        fi
        nohup "$PYTHON_BIN" -u "$WATCHER_PY" \
            --watch "$WATCH_DIR" \
            --remote "$GDRIVE_REMOTE" \
            --interval 30 \
            --max-existing "$MAX_EXISTING" \
            >> "$LOG_FILE" 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "$PID_FILE"
        echo "Started GDrive checkpoint watcher (PID $NEW_PID)"
        echo "  Watching: $WATCH_DIR"
        echo "  Remote:   $GDRIVE_REMOTE"
        echo "  Log:      $LOG_FILE"
        ;;
    stop)
        if is_running; then
            PID=$(cat "$PID_FILE")
            kill "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "Stopped GDrive watcher (PID $PID)."
        else
            echo "GDrive watcher is not running."
            rm -f "$PID_FILE"
        fi
        ;;
    status)
        if is_running; then
            echo "GDrive watcher: RUNNING (PID $(cat "$PID_FILE"))"
        else
            echo "GDrive watcher: STOPPED"
        fi
        if command -v rclone >/dev/null 2>&1; then
            echo "rclone: $(command -v rclone) ($(rclone version | head -n 1))"
            echo "Configured remotes:"
            rclone listremotes || true
        else
            echo "rclone: NOT FOUND in PATH"
        fi
        echo "Recent watcher log ($LOG_FILE):"
        tail -n 10 "$LOG_FILE" 2>/dev/null || echo "  (no log yet)"
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {start [--all-existing]|stop|status|logs}"
        exit 1
        ;;
esac

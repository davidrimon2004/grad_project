#!/usr/bin/env bash
# One-time interactive Google Drive authorization & always-on uploader setup.
# Run on this SSH host:
#   bash scripts/setup_gdrive.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_NAME="${GDRIVE_REMOTE_NAME:-gdrive}"
REMOTE_ROOT="${GDRIVE_REMOTE:-${REMOTE_NAME}:Video-LLaVA/checkpoints}"

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is not installed in PATH or ~/.local/bin."
    exit 1
fi

echo "============================================================"
echo "  Google Drive Auto-Upload Setup (rclone)"
echo "============================================================"
echo "rclone binary: $(command -v rclone)"
rclone version | head -n 1
echo

HAS_TOKEN=0
if rclone listremotes | grep -qx "${REMOTE_NAME}:"; then
    if rclone config show "${REMOTE_NAME}" 2>/dev/null | grep -qE "^(token|service_account_file) *= *.+"; then
        HAS_TOKEN=1
    fi
fi

if [ "$HAS_TOKEN" -eq 1 ]; then
    echo "Remote '${REMOTE_NAME}:' is ALREADY configured with credentials!"
else
    if ! rclone listremotes | grep -qx "${REMOTE_NAME}:"; then
        rclone config create "${REMOTE_NAME}" drive scope=drive --non-interactive >/dev/null 2>&1 || true
    else
        rclone config update "${REMOTE_NAME}" scope=drive --non-interactive >/dev/null 2>&1 || true
    fi

    echo "Starting interactive Google Drive OAuth authorization for '${REMOTE_NAME}:'..."
    echo ""
    echo "Choose whichever is easiest for you:"
    echo "  Option A (easiest if your SSH/IDE forwards localhost ports — no local rclone needed):"
    echo "    - Answer 'y' to 'Use web browser to automatically authenticate?'"
    echo "    - Click/open the printed http://127.0.0.1:53682/auth?... link in your local browser."
    echo ""
    echo "  Option B (if port 53682 is NOT forwarded — standard headless SSH):"
    echo "    - Answer 'n' to 'Use web browser to automatically authenticate?'"
    echo "    - Run the printed 'rclone authorize \"drive\" ...' command on your local PC,"
    echo "      sign in to Google Drive, and paste the resulting token back here."
    echo ""
    echo "  (When asked 'Configure as Shared Drive?' -> answer 'n')"
    echo ""
    rclone config reconnect "${REMOTE_NAME}:"
fi

echo ""
echo "Testing connection to ${REMOTE_NAME}: ..."
rclone mkdir "${REMOTE_ROOT}"
rclone lsd "${REMOTE_NAME}:Video-LLaVA"
echo "SUCCESS: Google Drive remote '${REMOTE_ROOT}' is connected and writable!"

# Try to install & enable a systemd user service so the watcher survives reboots
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
if command -v systemctl >/dev/null 2>&1 && systemctl --user status >/dev/null 2>&1; then
    mkdir -p "$SYSTEMD_USER_DIR"
    PYTHON_BIN="python3"
    if [ -x "$HOME/miniconda3/envs/videollava/bin/python" ]; then
        PYTHON_BIN="$HOME/miniconda3/envs/videollava/bin/python"
    fi
    cat > "${SYSTEMD_USER_DIR}/videollava-gdrive-watcher.service" <<EOF
[Unit]
Description=Video-LLaVA Google Drive Checkpoint Auto-Uploader
After=network-online.target

[Service]
Type=simple
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${PYTHON_BIN} -u ${SCRIPT_DIR}/gdrive_checkpoint_watcher.py --watch $(cd "${SCRIPT_DIR}/.." && pwd)/checkpoints --remote ${REMOTE_ROOT} --interval 30 --max-existing 1
Restart=always
RestartSec=15
StandardOutput=append:%h/gdrive_upload_watcher.log
StandardError=append:%h/gdrive_upload_watcher.log

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now videollava-gdrive-watcher.service
    echo "Installed and started systemd user service: videollava-gdrive-watcher.service"
    echo "Check status anytime with: systemctl --user status videollava-gdrive-watcher"
else
    bash "${SCRIPT_DIR}/start_gdrive_watcher.sh" start
fi

echo ""
echo "All set! Every completed checkpoint will now automatically upload to:"
echo "  ${REMOTE_ROOT}/<run_name>/checkpoint-<step>/"
echo "And final weights will upload to:"
echo "  ${REMOTE_ROOT}/<run_name>/final/"

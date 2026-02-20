#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user@server> [tag] [target_dir] [ssh_key]"
  exit 1
fi

SERVER="$1"
TAG="${2:-$(date +%Y.%m.%d-%H%M)}"
TARGET_DIR="${3:-/home/user/GROM/ostrofun}"
SSH_KEY="${4:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_IMAGE="ostrofun-backend:${TAG}"
FRONTEND_IMAGE="ostrofun-frontend:${TAG}"
ARCHIVE_NAME="ostrofun-images-${TAG}.tar"

echo "[1/4] Build backend image: ${BACKEND_IMAGE}"
docker build -f production_dev/Dockerfile -t "${BACKEND_IMAGE}" .

echo "[2/4] Build frontend image: ${FRONTEND_IMAGE}"
docker build -f production_dev/frontend/Dockerfile -t "${FRONTEND_IMAGE}" .

echo "[3/4] Save images to archive: ${ARCHIVE_NAME}"
docker save -o "${ARCHIVE_NAME}" "${BACKEND_IMAGE}" "${FRONTEND_IMAGE}"

SCP_CMD=(scp -r)
if [[ -n "${SSH_KEY}" ]]; then
  SCP_CMD+=(-i "${SSH_KEY}")
fi

echo "[4/4] Copy release files to ${SERVER}:${TARGET_DIR}"
"${SCP_CMD[@]}" \
  production_dev/docker-compose.deploy.yml \
  production_dev/.env.deploy.example \
  production_dev/server_apply_release.sh \
  production_dev/security \
  "${ARCHIVE_NAME}" \
  "${SERVER}:${TARGET_DIR}/"

echo
echo "Done."
echo "Run on server:"
echo "ssh ${SERVER}"
echo "cd ${TARGET_DIR}"
echo "chmod +x ./server_apply_release.sh"
echo "./server_apply_release.sh ${TAG}"

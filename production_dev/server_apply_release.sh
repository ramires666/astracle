#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tag> [target_dir]"
  exit 1
fi

TAG="$1"
TARGET_DIR="${2:-/home/user/GROM/ostrofun}"
ARCHIVE_NAME="ostrofun-images-${TAG}.tar"
ENV_FILE="${TARGET_DIR}/.env"
COMPOSE_FILE="${TARGET_DIR}/docker-compose.deploy.yml"

set_env_value() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    sed -i "s#^${key}=.*#${key}=${value}#g" "${ENV_FILE}"
  else
    printf "%s=%s\n" "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

cd "${TARGET_DIR}"

if [[ ! -f "${ARCHIVE_NAME}" ]]; then
  echo "Error: archive not found: ${TARGET_DIR}/${ARCHIVE_NAME}"
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: compose file not found: ${COMPOSE_FILE}"
  exit 1
fi

mkdir -p storage/market storage/prediction_cache storage/research_cache
mkdir -p security/nginx-logs security/fail2ban
touch "${ENV_FILE}"

echo "[1/3] Load docker archive: ${ARCHIVE_NAME}"
docker load -i "${ARCHIVE_NAME}"

echo "[2/3] Update .env image tags"
set_env_value "BACKEND_IMAGE" "ostrofun-backend:${TAG}"
set_env_value "FRONTEND_IMAGE" "ostrofun-frontend:${TAG}"
set_env_value "PUBLIC_PORT" "9742"
set_env_value "STORAGE_ROOT" "./storage"
set_env_value "LIVE_REFRESH_ENABLED" "1"
set_env_value "LIVE_REFRESH_INTERVAL_SECONDS" "30"
set_env_value "LIVE_REFRESH_PRICE_MOVE_THRESHOLD" "0.03"

echo "[3/3] Start services"
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" ps

echo
echo "Health check:"
curl -fsS http://localhost:9742/api/health || true
echo

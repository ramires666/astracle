# Ostrofun Production Deploy

This guide covers:
- image build on Windows and Linux
- image transfer to Linux server
- deploy from `/home/user/GROM/ostrofun`
- persistent storage for market/cache data

Deployment compose file:
- `production_dev/docker-compose.deploy.yml`

## 1) Target Folder On Linux Server

Use exactly this folder:

```text
/home/user/GROM/ostrofun
|-- docker-compose.deploy.yml
|-- .env
|-- ostrofun-images-<TAG>.tar
`-- storage
    |-- market
    |-- prediction_cache
    `-- research_cache
```

Persistent mounts:
- `./storage/market` -> `/app/data/market`
- `./storage/prediction_cache` -> `/app/data/prediction_cache`
- `./storage/research_cache` -> `/app/RESEARCH/cache`

## 2) Build Image On Windows (PowerShell)

Run from repository root:

```powershell
$TAG = "2026.02.19"
docker build -f production_dev/Dockerfile -t ostrofun-backend:$TAG .
docker build -f production_dev/frontend/Dockerfile -t ostrofun-frontend:$TAG .
docker save -o ostrofun-images-$TAG.tar ostrofun-backend:$TAG ostrofun-frontend:$TAG
```

## 3) Build Image On Linux (Bash)

Run from repository root:

```bash
TAG="2026.02.19"
docker build -f production_dev/Dockerfile -t ostrofun-backend:$TAG .
docker build -f production_dev/frontend/Dockerfile -t ostrofun-frontend:$TAG .
docker save -o ostrofun-images-$TAG.tar ostrofun-backend:$TAG ostrofun-frontend:$TAG
```

## 4) Copy To Production Server

From build machine:

```bash
scp production_dev/docker-compose.deploy.yml user@SERVER_IP:/home/user/GROM/ostrofun/
scp production_dev/.env.deploy.example user@SERVER_IP:/home/user/GROM/ostrofun/.env
scp ostrofun-images-2026.02.19.tar user@SERVER_IP:/home/user/GROM/ostrofun/
```

### 4.1) Copy From Windows (PowerShell + OpenSSH)

Run in PowerShell from repository root:

```powershell
$SERVER="user@SERVER_IP"
$TARGET="/home/user/GROM/ostrofun"
$TAG="2026.02.19"

# Optional: if you use private key auth
# $KEY="C:\Users\admin\.ssh\id_ed25519"

scp .\production_dev\docker-compose.deploy.yml "$SERVER:$TARGET/"
scp .\production_dev\.env.deploy.example "$SERVER:$TARGET/.env"
scp .\ostrofun-images-$TAG.tar "$SERVER:$TARGET/"

# With explicit key:
# scp -i $KEY .\ostrofun-images-$TAG.tar "$SERVER:$TARGET/"
```

If `scp` command is missing on Windows, install OpenSSH Client:
- `Settings -> Apps -> Optional features -> Add feature -> OpenSSH Client`
- then reopen PowerShell and repeat commands above.

## 5) First Start On Production Server

```bash
ssh user@SERVER_IP
cd /home/user/GROM/ostrofun
mkdir -p storage/market storage/prediction_cache storage/research_cache
docker load -i ostrofun-images-2026.02.19.tar
```

Edit `.env`:

```dotenv
BACKEND_IMAGE=ostrofun-backend:2026.02.19
FRONTEND_IMAGE=ostrofun-frontend:2026.02.19
PUBLIC_PORT=9742
STORAGE_ROOT=./storage
ALLOW_LEGACY_MODEL_FALLBACK=0
LIVE_REFRESH_ENABLED=1
LIVE_REFRESH_INTERVAL_SECONDS=3600
LIVE_REFRESH_PRICE_MOVE_THRESHOLD=0.03
MARKET_UPDATE_ALLOW_BINANCE_FALLBACK=0
COINGECKO=
```

Start:

```bash
docker compose -f docker-compose.deploy.yml --env-file .env up -d
```

## 6) Verify

```bash
cd /home/user/GROM/ostrofun
docker compose -f docker-compose.deploy.yml --env-file .env ps
curl http://localhost:9742/api/health
curl http://localhost:9742/api/refresh/status
```

Web UI:
- `http://SERVER_IP:9742`

## 7) How Persistent Storage Works

- Service does not require a SQL database for this API flow.
- Market and prediction cache are file-based and stored in `./storage/*`.
- On very first run with empty storage, container seeds `market` and `prediction_cache` from built-in image snapshots.
- After that, live refresh updates files in persistent folders, so data survives container recreation.
- `research_cache` is also persistent to avoid repeated heavy feature recalculation.

## 8) Update Release

Build and copy new tar, then on server:

```bash
cd /home/user/GROM/ostrofun
docker load -i ostrofun-images-NEW_TAG.tar
```

Update tags in `.env`, then:

```bash
docker compose -f docker-compose.deploy.yml --env-file .env up -d
```

Data in `./storage/*` remains intact.

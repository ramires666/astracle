# Deploy Without Registry (Simple Flow)

This flow is for one production server and no Docker registry.

Pipeline:
- build images locally
- save to one tar archive
- copy files to server via `scp`
- load archive on server and run compose

Target server folder:
- `/home/user/GROM/ostrofun`

## Files

- Local Windows script: `production_dev/deploy_no_registry.ps1`
- Local Linux script: `production_dev/deploy_no_registry.sh`
- Server apply script: `production_dev/server_apply_release.sh`
- Compose file: `production_dev/docker-compose.deploy.yml`

## 1) Windows (PowerShell)

Run from repository root:

```powershell
.\production_dev\deploy_no_registry.ps1 -Server "prod" -Tag "2026.02.20-001"
```

Optional SSH key:

```powershell
.\production_dev\deploy_no_registry.ps1 -Server "user@SERVER_IP" -Tag "2026.02.20-001" -SshKey "C:\Users\admin\.ssh\id_ed25519"
```

Notes:
- `-Server` can be SSH alias from your `~/.ssh/config` (for example `prod`).
- Script uses one `scp` command for all files, so auth prompt appears only once.

## 2) Linux/macOS (Bash)

Run from repository root:

```bash
chmod +x production_dev/deploy_no_registry.sh
./production_dev/deploy_no_registry.sh prod 2026.02.20-001
```

With SSH key:

```bash
./production_dev/deploy_no_registry.sh user@SERVER_IP 2026.02.20-001 /home/user/GROM/ostrofun ~/.ssh/id_ed25519
```

Notes:
- `SERVER` can be SSH alias (`prod`) from `~/.ssh/config`.
- Script uses one `scp` command for all files, so auth prompt appears only once.

## 3) Apply Release On Server

```bash
ssh user@SERVER_IP
cd /home/user/GROM/ostrofun
chmod +x ./server_apply_release.sh
./server_apply_release.sh 2026.02.20-001
```

What the server script does:
- `docker load -i ostrofun-images-<TAG>.tar`
- updates `.env` image tags
- ensures persistent storage folders exist
- ensures `security/nginx-logs` exists for frontend logs
- sets live refresh interval to `30` seconds
- runs `docker compose up -d`

## 4) Verify

```bash
cd /home/user/GROM/ostrofun
docker compose -f docker-compose.deploy.yml --env-file .env ps
curl http://localhost:9742/api/health
curl http://localhost:9742/api/refresh/status
```

## 5) Enable Fail2Ban (Before Frontend)

Install and enable Fail2Ban on production host:

```bash
sudo apt-get update
sudo apt-get install -y fail2ban
sudo cp /home/user/GROM/ostrofun/security/fail2ban/jail.local /etc/fail2ban/jail.d/ostrofun.local
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban
sudo fail2ban-client status
```

Detailed notes:
- `production_dev/security/fail2ban/README.md`

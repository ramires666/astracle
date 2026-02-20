# Self-Hosted Private Docker Registry (On-Prem)

This guide explains how to run a private Docker registry on your Linux `prod` server and expose it securely.

Registry base path:
- `/srv/REGISTRY`

Project deploy path (application stack):
- `/home/user/GROM/ostrofun`

Main options:
- `Option A`: registry + Caddy (TLS in the same stack)
- `Option B`: registry only on `prod`, TLS/reverse proxy on an existing external Nginx server

If you already have a separate Nginx server, use:
- `Section 14) Existing External Nginx (No Caddy on prod)`

## 1) Prerequisites

On `prod`:
- Docker Engine installed
- Docker Compose plugin installed (`docker compose version`)
- Domain name for registry, for example `registry.example.com`

Check:

```bash
docker --version
docker compose version
```

## 2) Network and Ports

Public inbound:
- `22/tcp` for SSH (prefer allowlist by source IP)
- `80/tcp` for ACME HTTP challenge and redirect
- `443/tcp` for HTTPS registry traffic

Do not expose publicly:
- `5000/tcp` (keep internal or restrict to trusted source only)

Example with UFW:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5000/tcp
sudo ufw enable
sudo ufw status verbose
```

If you use cloud firewalls/security groups, mirror the same policy there.

## 3) Create Registry Folders

```bash
sudo mkdir -p /srv/REGISTRY/{data,auth,caddy_data,caddy_config}
sudo chown -R $USER:$USER /srv/REGISTRY
cd /srv/REGISTRY
```

Expected structure:

```text
/srv/REGISTRY
|-- docker-compose.registry.yml
|-- Caddyfile
|-- auth/htpasswd
|-- data/
|-- caddy_data/
`-- caddy_config/
```

## 4) Create Registry Credentials

Replace password with a strong one:

```bash
cd /srv/REGISTRY
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn registry_admin 'ChangeThisStrongPassword!' > auth/htpasswd
chmod 640 auth/htpasswd
```

## 5) Create `docker-compose.registry.yml` (Registry + Caddy)

File: `/srv/REGISTRY/docker-compose.registry.yml`

```yaml
services:
  registry:
    image: registry:2.8.3
    container_name: registry
    restart: unless-stopped
    environment:
      REGISTRY_HTTP_ADDR: 0.0.0.0:5000
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_STORAGE_DELETE_ENABLED: "true"
    volumes:
      - ./data:/var/lib/registry
      - ./auth:/auth:ro
    networks:
      - registry_net

  caddy:
    image: caddy:2.9-alpine
    container_name: registry-caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./caddy_data:/data
      - ./caddy_config:/config
    depends_on:
      - registry
    networks:
      - registry_net

networks:
  registry_net:
    name: registry_net
```

## 6) Create `Caddyfile`

File: `/srv/REGISTRY/Caddyfile`

```caddy
registry.example.com {
    encode zstd gzip
    reverse_proxy registry:5000
}
```

Replace `registry.example.com` with your domain.

Optional IP allowlist:

```caddy
registry.example.com {
    @blocked not remote_ip 203.0.113.10/32 198.51.100.0/24
    respond @blocked 403
    reverse_proxy registry:5000
}
```

## 7) Start Registry Stack

```bash
cd /srv/REGISTRY
docker compose -f docker-compose.registry.yml up -d
docker compose -f docker-compose.registry.yml ps
docker compose -f docker-compose.registry.yml logs -f caddy
```

## 8) Verify HTTPS and Auth

From any machine:

```bash
curl -I https://registry.example.com/v2/
```

Expected: `401 Unauthorized` (correct, registry requests auth).

Authenticated check:

```bash
curl -u registry_admin:'ChangeThisStrongPassword!' https://registry.example.com/v2/_catalog
```

Expected:

```json
{"repositories":[]}
```

## 9) Push Images

### 9.1 Linux/macOS

```bash
REG=registry.example.com
TAG=2026.02.19

docker login $REG
docker tag ostrofun-backend:$TAG $REG/ostrofun/backend:$TAG
docker tag ostrofun-frontend:$TAG $REG/ostrofun/frontend:$TAG
docker push $REG/ostrofun/backend:$TAG
docker push $REG/ostrofun/frontend:$TAG
```

### 9.2 Windows (PowerShell)

```powershell
$REG = "registry.example.com"
$TAG = "2026.02.19"

docker login $REG
docker tag ostrofun-backend:$TAG "$REG/ostrofun/backend:$TAG"
docker tag ostrofun-frontend:$TAG "$REG/ostrofun/frontend:$TAG"
docker push "$REG/ostrofun/backend:$TAG"
docker push "$REG/ostrofun/frontend:$TAG"
```

## 10) Run Containers From Registry on `prod`

Login once on `prod`:

```bash
docker login registry.example.com
```

Update `/home/user/GROM/ostrofun/.env`:

```dotenv
BACKEND_IMAGE=registry.example.com/ostrofun/backend:2026.02.19
FRONTEND_IMAGE=registry.example.com/ostrofun/frontend:2026.02.19
PUBLIC_PORT=9742
STORAGE_ROOT=./storage
```

Deploy:

```bash
cd /home/user/GROM/ostrofun
docker compose -f docker-compose.deploy.yml --env-file .env pull
docker compose -f docker-compose.deploy.yml --env-file .env up -d
docker compose -f docker-compose.deploy.yml --env-file .env ps
```

## 11) Security Checklist

- Use domain + HTTPS only.
- Keep `5000` closed from public Internet.
- Use strong auth password in `auth/htpasswd`.
- Restrict SSH by source IP.
- Optionally add IP allowlist.
- Keep Docker/Caddy/OS updated.
- Back up `/srv/REGISTRY/data`.

## 12) Maintenance

List repositories and tags:

```bash
curl -u registry_admin:'ChangeThisStrongPassword!' https://registry.example.com/v2/_catalog
curl -u registry_admin:'ChangeThisStrongPassword!' https://registry.example.com/v2/ostrofun/backend/tags/list
```

Restart stack:

```bash
cd /srv/REGISTRY
docker compose -f docker-compose.registry.yml restart
```

Backup:

```bash
tar -czf /srv/registry-backup-$(date +%F).tar.gz /srv/REGISTRY/data
```

## 13) Quick Internal Test (No TLS)

For temporary LAN-only tests:
- expose `5000:5000` directly
- configure Docker clients with `insecure-registries`

Example `/etc/docker/daemon.json`:

```json
{
  "insecure-registries": ["SERVER_IP:5000"]
}
```

Restart Docker daemon after change. Do not use this mode on public Internet.

## 14) Existing External Nginx (No Caddy on prod)

Use this if you already have a dedicated Nginx server.

Architecture:
- `prod` runs only `registry:2` on port `5000`
- external Nginx terminates TLS and proxies to `prod:5000`
- clients use `https://registry.example.com`

### 14.1 Registry on `prod`

```bash
sudo mkdir -p /srv/REGISTRY/{data,auth}
sudo chown -R $USER:$USER /srv/REGISTRY
cd /srv/REGISTRY
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn registry_admin 'ChangeThisStrongPassword!' > auth/htpasswd
chmod 640 auth/htpasswd
```

File: `/srv/REGISTRY/docker-compose.registry.yml`

```yaml
services:
  registry:
    image: registry:2.8.3
    container_name: registry
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      REGISTRY_HTTP_ADDR: 0.0.0.0:5000
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_STORAGE_DELETE_ENABLED: "true"
    volumes:
      - ./data:/var/lib/registry
      - ./auth:/auth:ro
```

Start:

```bash
cd /srv/REGISTRY
docker compose -f docker-compose.registry.yml up -d
docker compose -f docker-compose.registry.yml ps
```

Firewall on `prod`:
- allow `5000/tcp` only from Nginx server IP
- deny `5000/tcp` for all other sources

UFW example:

```bash
sudo ufw allow from <NGINX_SERVER_IP> to any port 5000 proto tcp
sudo ufw deny 5000/tcp
sudo ufw status numbered
```

### 14.2 Nginx Config on Separate Nginx Server

File example: `/etc/nginx/sites-available/registry.conf`

```nginx
upstream docker_registry_backend {
    server <PROD_SERVER_IP>:5000;
    keepalive 32;
}

server {
    listen 80;
    server_name registry.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name registry.example.com;

    ssl_certificate     /etc/letsencrypt/live/registry.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/registry.example.com/privkey.pem;

    client_max_body_size 0;
    chunked_transfer_encoding on;

    location /v2/ {
        proxy_pass http://docker_registry_backend;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 900;
        proxy_send_timeout 900;
        proxy_buffering off;
        proxy_request_buffering off;
        add_header Docker-Distribution-Api-Version registry/2.0 always;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/registry.conf /etc/nginx/sites-enabled/registry.conf
sudo nginx -t
sudo systemctl reload nginx
```

Check:

```bash
curl -I https://registry.example.com/v2/
```

Expected: `401 Unauthorized`.

### 14.3 Push/Pull in Nginx Variant

From client:

```bash
docker login registry.example.com
docker tag ostrofun-backend:2026.02.19 registry.example.com/ostrofun/backend:2026.02.19
docker push registry.example.com/ostrofun/backend:2026.02.19
```

On `prod` deployment host:

```bash
docker login registry.example.com
cd /home/user/GROM/ostrofun
docker compose -f docker-compose.deploy.yml --env-file .env pull
docker compose -f docker-compose.deploy.yml --env-file .env up -d
```

### 14.4 Auth Placement Recommendation

Recommended:
- keep auth inside `registry:2` (`htpasswd`)
- use Nginx only for TLS and reverse proxy

This avoids auth duplication and keeps credentials in one place.

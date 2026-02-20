# Docker Registry Quickstart (Self-Hosted, Project-Agnostic)

This is a short setup for a private Docker registry on Linux.

## 1) Prerequisites

- Docker installed
- Docker Compose plugin installed

Check:

```bash
docker --version
docker compose version
```

## 2) Create folders

```bash
sudo mkdir -p /srv/REGISTRY/{data,auth}
sudo chown -R $USER:$USER /srv/REGISTRY
cd /srv/REGISTRY
```

## 3) Create username/password

```bash
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn registry_admin 'ChangeThisStrongPassword!' > auth/htpasswd
chmod 640 auth/htpasswd
```

## 4) Create `docker-compose.yml`

File: `/srv/REGISTRY/docker-compose.yml`

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

## 5) Start registry

```bash
cd /srv/REGISTRY
docker compose up -d
docker compose ps
```

## 6) Verify

```bash
curl -I http://SERVER_IP:5000/v2/
```

Expected: `401 Unauthorized` (this is correct).

## 7) Push and pull example

```bash
docker login SERVER_IP:5000
docker tag nginx:alpine SERVER_IP:5000/demo/nginx:1
docker push SERVER_IP:5000/demo/nginx:1
docker pull SERVER_IP:5000/demo/nginx:1
```

## 8) Security (important)

For Internet-facing use:
- put registry behind HTTPS reverse proxy (Nginx/Caddy)
- keep `5000/tcp` closed from public Internet
- allow `5000/tcp` only from reverse-proxy host

Quick UFW example on registry host:

```bash
sudo ufw allow from <REVERSE_PROXY_IP> to any port 5000 proto tcp
sudo ufw deny 5000/tcp
```

## 9) User management (`htpasswd`)

List users:

```bash
cut -d: -f1 /srv/REGISTRY/auth/htpasswd
```

Add new user:

```bash
docker run --rm -v /srv/REGISTRY/auth:/auth --entrypoint htpasswd httpd:2.4-alpine -Bb /auth/htpasswd new_user 'StrongPass123!'
docker compose -f /srv/REGISTRY/docker-compose.yml restart
```

Change password for existing user:

```bash
docker run --rm -v /srv/REGISTRY/auth:/auth --entrypoint htpasswd httpd:2.4-alpine -Bb /auth/htpasswd registry_admin 'NewStrongPass456!'
docker compose -f /srv/REGISTRY/docker-compose.yml restart
```

Delete user:

```bash
sudo sed -i '/^new_user:/d' /srv/REGISTRY/auth/htpasswd
docker compose -f /srv/REGISTRY/docker-compose.yml restart
```

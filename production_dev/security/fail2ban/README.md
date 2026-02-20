# Fail2Ban For Frontend Nginx

This setup protects the frontend edge (`btc-astro-frontend`) by banning abusive IPs
based on Nginx logs written to:

- `/home/user/GROM/ostrofun/security/nginx-logs/access.log`
- `/home/user/GROM/ostrofun/security/nginx-logs/error.log`

## 1) Install fail2ban on server

```bash
sudo apt-get update
sudo apt-get install -y fail2ban
```

## 2) Install jail config

```bash
sudo cp /home/user/GROM/ostrofun/security/fail2ban/jail.local /etc/fail2ban/jail.d/ostrofun.local
sudo chmod 644 /etc/fail2ban/jail.d/ostrofun.local
```

## 3) Enable and restart

```bash
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban
```

## 4) Verify

```bash
sudo fail2ban-client status
sudo fail2ban-client status ostrofun-nginx-botsearch
sudo fail2ban-client status ostrofun-nginx-limit-req
```

## Notes

- Frontend Nginx already has request limiting for `/api/`.
- If your host uses UFW and you want UFW ban actions, set `banaction = ufw`
  under `[DEFAULT]` in `jail.local`.
- If you use a separate edge reverse proxy (for example Nginx Proxy Manager on
  another server), install Fail2Ban on that edge host instead of app host.
  Otherwise bans may target only the proxy IP.

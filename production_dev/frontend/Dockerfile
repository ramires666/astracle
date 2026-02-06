FROM nginx:1.27-alpine

# Replace default site config with API reverse-proxy setup.
COPY production_dev/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Serve frontend as static files.
COPY production_dev/static/ /usr/share/nginx/html/static/
COPY production_dev/static/index.html /usr/share/nginx/html/index.html

EXPOSE 80

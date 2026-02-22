FROM docker.io/dxflrs/garage:v2.2.0 AS garage
FROM alpine:3.21
COPY --from=garage /garage /usr/local/bin/garage
RUN apk add --no-cache curl
COPY docker/garage-entrypoint.sh /usr/local/bin/garage-entrypoint.sh
RUN chmod +x /usr/local/bin/garage-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/garage-entrypoint.sh"]

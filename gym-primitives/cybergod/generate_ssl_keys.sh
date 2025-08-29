#!/bin/bash
DOMAIN="Cybergod Central Bank"

# Generate unencrypted private key
openssl genpkey -algorithm RSA -out key.pem -outform PEM

# Generate self-signed certificate with 100-year validity
openssl req -x509 -new -key private.key -out cert.pem -days 36500 \
  -subj "/C=US/ST=California/L=Palo Alto/O=Cybergod AGI Research/CN=$DOMAIN" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign"

echo "Unencrypted private key and certificate generated."
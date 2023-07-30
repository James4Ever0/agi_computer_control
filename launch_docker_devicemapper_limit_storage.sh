# --storage-opt is supported only for overlay over xfs with 'pquota' mount option.
# docker run  --storage-opt size=10M --rm -it alpine

# when using devmapper, make sure size is greater than 10G (default)
# https://docs.docker.com/storage/storagedriver/device-mapper-driver/#configure-direct-lvm-mode-for-production
docker run --storage-opt size=11G --rm -it alpine

# --storage-opt is supported only for overlay over xfs with 'pquota' mount option.
# change "data-root" to somewhere else in /etc/docker/daemon.json.
docker run  --storage-opt size=10M --rm -it alpine

# when using devmapper, make sure size is greater than 10G (default)
# https://docs.docker.com/storage/storagedriver/device-mapper-driver/#configure-direct-lvm-mode-for-production
# docker run --storage-opt size=11G --rm -it alpine

# zfs, vfs (not a unionfs, but for testing) supports disk quota.

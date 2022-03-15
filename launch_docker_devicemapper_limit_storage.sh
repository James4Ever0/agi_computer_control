# --storage-opt is supported only for overlay over xfs with 'pquota' mount option.
# change "data-root" to somewhere else in /etc/docker/daemon.json.
# edit /etc/fstab and add our xfs block on new line (find uuid using blkid)
docker run  --storage-opt size=10M --rm -it alpine

# when using devmapper, make sure size is greater than 10G (default)
# https://docs.docker.com/storage/storagedriver/device-mapper-driver/#configure-direct-lvm-mode-for-production
# docker run --storage-opt size=11G --rm -it alpine

# zfs, vfs (not a unionfs, but for testing) supports disk quota.

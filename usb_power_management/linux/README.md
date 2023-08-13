do not use 'laptop-mode' or 'powertop' or 'tlp'

## webdav

server (may restart on error):

```bash
# first mount the hard drive to some path.
rclone serve webdav /root
# add "-L" to follow symlinks
```

client (only need to start once when server is running):

```bash
# need "davfs2" package
mkdir /mnt/root_webdav
echo -e "\n\n" | mount -t davfs http://127.0.0.1:8080/ /mnt/root_webdav/
```

## auto mount external/removable drives (to fixed path)

disable 

write this to `/etc/fstab`:

```
UUID=XXXXXXXXXXXXXXX    /myhdd <fstype>  auto,nofail,noatime,rw,user    0   0
```

run this command every minute with crontab:

```bash
mount -a
```
do not use 'laptop-mode' or 'powertop' or 'tlp'

## hdparm

disable power management

```bash
hdparm -B 255 /dev/sd<n>
```

## dirty bytes

```bash
echo $((16*1024*1024)) > /proc/sys/vm/dirty_bytes
echo $((48*1024*1024)) > /proc/sys/vm/dirty_background_bytes
```

## sync

when trying to activate external disks in regular intervals, remember to sync changes to devices:

```bash
touch <mountpoint>/
```

## webdav

server (may manually/explicitly restart on reconnection):

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

check if automount is enabled

```bash
gsettings list-recursively org.gnome.desktop.media-handling 
```

disable gnome automount:

```bash
gsettings set org.gnome.desktop.media-handling automount true
gsettings set org.gnome.desktop.media-handling automount-open true
```

write this to `/etc/fstab`:

```
UUID=XXXXXXXXXXXXXXX    /myhdd <fstype>  auto,nofail,noatime,rw,user    0   0
```

run this command every minute with crontab:

```bash
mount -a
```

## reset usb

under `/sys/bus/pci/drivers/(xhci_hci|ehci)`, echo usb device ids (once at a time) to `unbind` and `bind` respectively to reset usb devices.
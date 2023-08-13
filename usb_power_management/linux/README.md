do not use 'laptop-mode' or 'powertop' or 'tlp'

## mountpoint

call `rmdir <mountpoint>` for safer removal.

if the program is still running and writing to mountpoint but disk is not available, then file will be created under that mountpoint, making it impossible to remount the disk to the same location (non-empty) unless remove that mountpoint by force.

## uas versus usb-storage

note: doing this might not help much, since we have this kind of disk detaching issue even with usb-storage driver (might be insufficient power?)

----

although uas (usb attached scsi) is faster than usb-storage (usb bot (bulk-only transport)), it is not stable. a bad standard specification and massive derivatives from vendors result into compatibility problems with linux kernel.

[issues on uas driver](https://forums.linuxmint.com/viewtopic.php?t=320801)

```bash
# method 1: only blacklist those unsupported uas devices
###########################################################
# look for bus number and device number of which using uas driver
lsusb -t
# get device id
lsusb
# write those device id to rule
echo "options usb-storage quirks=174c:2364:u,152d:0583:u" > /etc/modprobe.d/blacklist-uas-on-quirks.conf

# method 2: disable uas kernel module entirely (not working?)
###########################################################
# get module dependencies
modinfo uas
# check if uas is not builtin (maybe we should just delete that damn kernel module (not working?))
find /lib/modules/$(uname -r) | grep uas
# not built-in, but dynamically loaded
echo "blacklist uas" >> /etc/modprobe.d/blacklist.conf
# after reboot, check if uas is disabled.
lsmod | grep uas

# 
```

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
touch <mountpoint>/.keepalive; sync; rm -f <mountpoint>/.keepalive; sync
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

the same for `/sys/bus/usb/drivers/*`.

find device id by "lsusb". use command "usbreset" from package "usbutils".
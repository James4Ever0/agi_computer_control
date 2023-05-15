cd agi_computer_control/ubuntu_qemu_utm_arm_record

# enable sudo
echo -e "KSDA37287522\n" | sudo -S echo "enable sudo"

# remove davfs pid locks
sudo rm /var/run/mount.davfs/mnt-dav1.pid
sudo rm /var/run/mount.davfs/mnt-dav2.pid

sudo apt install -y davfs2
#sudo mkdir /mnt/dav1
sudo mkdir /mnt/dav2

sudo umount /mnt/dav1
sudo umount /mnt/dav2

sudo rm -rf scripts
sudo rm -rf recordings

#echo -e "root\nroot\n" | sudo mount -t davfs -o noexec http://192.168.56.1:8110 /mnt/dav1
echo -e "root\nroot\n" | sudo mount -t davfs -o noexec http://192.168.56.1:8111 /mnt/dav2

#mkdir scripts
#mkdir recordings
#ln -s /mnt/dav1 scripts
ln -s /mnt/dav2 recordings




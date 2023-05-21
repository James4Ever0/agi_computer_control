echo -e "110110\n" | sudo -S echo "enable sudo"
sudo apt install -y davfs2

sudo umount /mnt/dav1
sudo umount /mnt/dav2

sudo rm -rf /mnt/dav1
sudo rm -rf /mnt/dav2

sudo mkdir /mnt/dav1
sudo mkdir /mnt/dav2
echo -e "root\nroot\n" | sudo mount -t davfs -o noexec http://10.0.1.6:8110 /mnt/dav1
echo -e "root\nroot\n" | sudo mount -t davfs -o noexec http://10.0.1.6:8111 /mnt/dav2

mkdir scripts
mkdir recordings

ln -s /mnt/dav1 scripts
ln -s /mnt/dav2 recordings
# figure out what version of debian is for kali 2022.2
# `uname -a` to get debian kernel version (5.16) -> bullseye (debian 11)

# supported versions:
# ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
distribution="debian11" \
      && curl -fsSL -k https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L -k https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
timeout 10 vboxmanage controlvm "12c0e77b-5f4a-4d30-b19b-1b105d2042cf" poweroff # Ubuntu 16.04

# 掉盘了 一般是
# 就没有更好的固态硬盘盒么

if [ "$?" -ne 0 ]; then
echo "ERROR: Failed to stop vm."
ps aux | grep -i /usr/lib/virtualbox | grep -v grep | awk '{print $2}' | xargs -iabc kill -s TERM abc
fi

# start in headless mode
# you may not live stream this one! but vrde is available! and you can use ffmpeg for bridging!
# vboxmanage startvm "Ubuntu 16.04" --type headless
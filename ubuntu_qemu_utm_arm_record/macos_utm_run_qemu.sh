# https://vaibhavkaushal.com/posts/ubuntu-focal-on-apple-silicon-m1/#step-4---launch-acvm

# EFI_PATH="/Volumes/Toshiba XG3/works/agi_computer_control/ubuntu_qemu_utm_arm_record/EFI/QEMU_EFI.fd"
EFI_PATH="/Volumes/Toshiba XG3/works/agi_computer_control/ubuntu_qemu_utm_arm_record/EFI/efi_vars.fd"
IMAGE_PATH="/Volumes/Toshiba XG3/UTM VMs/Ubuntu ARM.utm/Data/335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8.qcow2"

qemu-system-aarch64 \
  -serial stdio \
  -M virt,highmem=off \
  -accel hvf \
  -cpu cortex-a72 \
  -smp 4,cores=4 \
  -m 2048 \
  -bios "$EFI_PATH" \
  -device virtio-gpu-pci \
  -display default,show-cursor=on \
  -device qemu-xhci \
  -device usb-kbd \
  -device usb-tablet \
  -device intel-hda \
  -device hda-duplex \
  -drive file="$IMAGE_PATH",if=virtio,cache=writethrough


# /Applications/UTM.app/Contents/XPCServices/QEMUHelper.xpc/Contents/MacOS/QEMULauncher.app/Contents/MacOS/QEMULauncher /Applications/UTM.app/Contents/Frameworks/qemu-aarch64-softmmu.framework/Versions/A/qemu-aarch64-softmmu -L /Applications/UTM.app/Contents/Resources/qemu -S -spice unix=on,addr=/Users/jamesbrown/Library/Group Containers/WDNLXAD4W8.com.utmapp.UTM/B54F0831-B34F-404D-B600-5E2886D8AC53.spice,disable-ticketing=on,image-compression=off,playback-compression=off,streaming-video=off,gl=off -chardev spiceport,id=org.qemu.monitor.qmp,name=org.qemu.monitor.qmp.0 -mon chardev=org.qemu.monitor.qmp,mode=control -nodefaults -vga none -device virtio-net-pci,mac=9A:B7:7D:21:25:67,netdev=net0 -netdev vmnet-shared,id=net0 -device virtio-ramfb -cpu host -smp cpus=4,sockets=1,cores=4,threads=1 -machine virt -accel hvf -drive if=pflash,format=raw,unit=0,file=/Applications/UTM.app/Contents/Resources/qemu/edk2-aarch64-code.fd,readonly=on -drive if=pflash,unit=1,file=/Volumes/Toshiba XG3/UTM VMs/Ubuntu ARM.utm/Data/efi_vars.fd -m 2048 -device intel-hda -device hda-duplex -device nec-usb-xhci,id=usb-bus -device usb-tablet,bus=usb-bus.0 -device usb-mouse,bus=usb-bus.0 -device usb-kbd,bus=usb-bus.0 -device qemu-xhci,id=usb-controller-0 -chardev spicevmc,name=usbredir,id=usbredirchardev0 -device usb-redir,chardev=usbredirchardev0,id=usbredirdev0,bus=usb-controller-0.0 -chardev spicevmc,name=usbredir,id=usbredirchardev1 -device usb-redir,chardev=usbredirchardev1,id=usbredirdev1,bus=usb-controller-0.0 -chardev spicevmc,name=usbredir,id=usbredirchardev2 -device usb-redir,chardev=usbredirchardev2,id=usbredirdev2,bus=usb-controller-0.0 -device usb-storage,drive=drive18D1A7CE-50FE-4567-B033-B2AB9F811108,removable=true,bootindex=0,bus=usb-bus.0 -drive if=none,media=cdrom,id=drive18D1A7CE-50FE-4567-B033-B2AB9F811108,readonly=on -device virtio-blk-pci,drive=drive335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8,bootindex=1 -drive if=none,media=disk,id=drive335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8,file=/Volumes/Toshiba XG3/UTM VMs/Ubuntu ARM.utm/Data/335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8.qcow2,discard=unmap,detect-zeroes=unmap -device virtio-serial -device virtserialport,chardev=vdagent,name=com.redhat.spice.0 -chardev spicevmc,id=vdagent,debug=0,name=vdagent -name Linux -uuid B54F0831-B34F-404D-B600-5E2886D8AC53 -device virtio-rng-pci

# qemu-system-aarch64 -L /Applications/UTM.app/Contents/Resources/qemu -S -spice unix=on,addr='/Users/jamesbrown/Library/Group Containers/WDNLXAD4W8.com.utmapp.UTM/B54F0831-B34F-404D-B600-5E2886D8AC53.spice',disable-ticketing=on,image-compression=off,playback-compression=off,streaming-video=off,gl=off -chardev spiceport,id=org.qemu.monitor.qmp,name=org.qemu.monitor.qmp.0 -mon chardev=org.qemu.monitor.qmp,mode=control -nodefaults -vga none -device virtio-net-pci,mac=9A:B7:7D:21:25:67,netdev=net0 -netdev vmnet-shared,id=net0 -device virtio-ramfb -cpu host -smp cpus=4,sockets=1,cores=4,threads=1 -machine virt -accel hvf -drive if=pflash,format=raw,unit=0,file='/Applications/UTM.app/Contents/Resources/qemu/edk2-aarch64-code.fd',readonly=on -drive if=pflash,unit=1,file='/Volumes/Toshiba XG3/UTM VMs/Ubuntu ARM.utm/Data/efi_vars.fd' -m 2048 -device intel-hda -device hda-duplex -device nec-usb-xhci,id=usb-bus -device usb-tablet,bus=usb-bus.0 -device usb-mouse,bus=usb-bus.0 -device usb-kbd,bus=usb-bus.0 -device qemu-xhci,id=usb-controller-0 -chardev spicevmc,name=usbredir,id=usbredirchardev0 -device usb-redir,chardev=usbredirchardev0,id=usbredirdev0,bus=usb-controller-0.0 -chardev spicevmc,name=usbredir,id=usbredirchardev1 -device usb-redir,chardev=usbredirchardev1,id=usbredirdev1,bus=usb-controller-0.0 -chardev spicevmc,name=usbredir,id=usbredirchardev2 -device usb-redir,chardev=usbredirchardev2,id=usbredirdev2,bus=usb-controller-0.0 -device usb-storage,drive=drive18D1A7CE-50FE-4567-B033-B2AB9F811108,removable=true,bootindex=0,bus=usb-bus.0 -drive if=none,media=cdrom,id=drive18D1A7CE-50FE-4567-B033-B2AB9F811108,readonly=on -device virtio-blk-pci,drive=drive335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8,bootindex=1 -drive if=none,media=disk,id=drive335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8,file='/Volumes/Toshiba XG3/UTM VMs/Ubuntu ARM.utm/Data/335D6651-2B4B-47A7-99B4-CBFB1EFDEFF8.qcow2',discard=unmap,detect-zeroes=unmap -device virtio-serial -device virtserialport,chardev=vdagent,name=com.redhat.spice.0 -chardev spicevmc,id=vdagent,debug=0,name=vdagent -name Linux -uuid B54F0831-B34F-404D-B600-5E2886D8AC53 -device virtio-rng-pci


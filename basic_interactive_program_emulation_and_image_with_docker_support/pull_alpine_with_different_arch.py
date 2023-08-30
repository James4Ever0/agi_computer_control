arch_list = ["amd64", "arm64", "arm/v7", "i386"]
target_image = "alpine:3.7"
target_fname_prefix = target_image.replace(":", "_")
import os

for arch in arch_list:
    fpath = f"{target_fname_prefix}_{arch.replace('/','')}.tar"
    cmds = [
        f'docker pull --platform "linux/{arch}" {target_image}',
        f"docker save {target_image} -o {fpath}",
        f"docker rmi {target_image}",
    ]
    for cmd in cmds:
        print("executing:", cmd)
        os.system(cmd)

from config import DOCKER_BIN, IS_DOCKER
import os

arch_list = ["amd64", "arm64", "arm/v7", "i386"]
target_image = "alpine:3.7" if IS_DOCKER else "docker.io/alpine:3.7"
target_fname_prefix = target_image.replace(":", "_")

dirpath = f"{DOCKER_BIN}_images"

# if os.path.exists(dirpath):
#     raise Exception(f"target directory '{dirpath}' already exists.")
os.makedirs(dirpath, exist_ok=True)

if __name__ == "__main__":
    for arch in arch_list:
        fpath = f"{target_fname_prefix}_{arch.replace('/','')}.tar"
        cmds = [
            f'{DOCKER_BIN} pull --platform "linux/{arch}" {target_image}',
            f"{DOCKER_BIN} save {target_image} -o {dirpath}/{fpath}",
            f"{DOCKER_BIN} rmi {target_image}",
        ]
        for cmd in cmds:
            print("executing:", cmd)
            os.system(cmd)

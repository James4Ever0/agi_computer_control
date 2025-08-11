# process steps:
# iterate over input dir
# read metadata, decide the input cast file path
# check if the gif file exists
# execute the conversion program agg and create gif
# copy the temporary gif to destination

import os
import json
import tempfile
import shutil

input_dir = "./recordings"
output_dir = "./gif_video"

print("Input dir:", input_dir)
print("Output dir:", output_dir)

agg_binary_path = "/home/jamesbrown/Downloads/agg-x86_64-unknown-linux-musl"

assert os.path.isdir(input_dir)

# check if the agg exists
if os.path.isfile(agg_binary_path):
    print("Using agg binary at:", agg_binary_path)
    ans = input("Proceed? (y/n) ")
    if ans.lower().strip() == "n":
        print("User aborting")
        exit(1)
    else:
        # make it executable, if the system is not windows
        if os.name != "nt":
            os.chmod(agg_binary_path, 0o755)
else:
    print("Agg does not exist at:", agg_binary_path)
    print("Aborting")

os.makedirs(output_dir, exist_ok=True)

# list the directories in input dir

record_ids = os.listdir(input_dir)

for it in record_ids:
    print("Processing record #%s" % it)
    info_filepath = os.path.join(input_dir, it, "info.json")
    if os.path.exists(info_filepath):
        with open(info_filepath, "r") as f:
            info_data = json.load(f)
            # get the extension name for the recording
            extension_name = info_data["asciicast_file_extension_name"]
            record_filepath = os.path.join(input_dir, it, "record{}".format(extension_name))
            if os.path.exists(record_filepath):
                # convert the record info gif
                gif_output_path = os.path.join(output_dir, "{}.gif".format(it))
                print("Output path:", gif_output_path)
                if os.path.isfile(gif_output_path):
                    print("Gif already exists for this record, skipping")
                    continue
                print("Converting record:", record_filepath)
                with tempfile.TemporaryDirectory() as tmpdir:
                # tmpdir = "/dev/shm"
                # if not os.path.exists(tmpdir):
                #     print("%s does not exist" % tmpdir)
                #     print("Stop looping")
                # else:
                    print("Using tmpdir:", tmpdir)
                    gif_tmp_output_path = os.path.join(tmpdir, "output.gif")
                    options = "--fps-cap 3 --renderer fontdue"
                    cmd = f"{agg_binary_path} {options} {record_filepath} {gif_tmp_output_path}"
                    print("Executing cmd:", cmd)
                    exitcode = os.system(cmd)
                    if exitcode == 130:
                        print("Keyboard interrupt")
                        print("Stop looping")
                        break
                    elif exitcode == 256:
                        print("Format parsing error")
                        print("Skipping")
                        continue
                    elif exitcode != 0:
                        print("Unknown non-zero exitcode:", exitcode)
                        print("Exiting")
                        break
                    # copy the tmp gif file to gif output path
                    print("Copying GIF to:", gif_output_path)
                    shutil.copy(gif_tmp_output_path, gif_output_path)
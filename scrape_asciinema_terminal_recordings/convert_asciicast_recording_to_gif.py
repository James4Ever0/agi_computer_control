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

# TODO: make it into multithreading

program_state = {"keyboard_interrupt":False}

def task_generator():
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
                    yield record_filepath, gif_output_path

def task_executor(record_filepath:str, gif_output_path:str):
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Using tmpdir:", tmpdir)
        gif_tmp_output_path = os.path.join(tmpdir, "output.gif")
        options = "--fps-cap 3 --renderer fontdue"
        cmd = f"{agg_binary_path} {options} {record_filepath} {gif_tmp_output_path}"
        print("Executing cmd:", cmd)
        exitcode = os.system(cmd)

        if exitcode == 130:
            print("Keyboard interrupt")
            print("Stop looping")
            program_state["keyboard_interrupt"] = True

        elif exitcode == 256:
            print("Format parsing error")
            print("Skipping")
    
        elif exitcode != 0:
            print("Unknown non-zero exitcode:", exitcode)

        # copy the tmp gif file to gif output path
        print("Copying GIF to:", gif_output_path)
        shutil.copy(gif_tmp_output_path, gif_output_path)
        return exitcode


def run_commands(max_workers=10):
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    """
    Execute shell commands from a generator in parallel.
    
    Args:
        max_workers: Maximum concurrent threads (default=10).
    """

    print("Starting executor with max_workers=%s" % max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_record_filepath = dict()
        for record_filepath, gif_output_path in task_generator():
            future_to_record_filepath[record_filepath] = executor.submit(task_executor, record_filepath, gif_output_path)
            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_record_filepath):
            record_filepath = future_to_record_filepath[future]
            try:
                return_code = future.result()
                assert return_code == 0
            except AssertionError: # non-fatal
                if program_state["keyboard_interrupt"]:
                    print("KeyboardInterrupt: User interrupted agg execution")
                    break # user interrupted
                else:
                    print("Agg process returned non-zero code:", return_code)
            except KeyboardInterrupt: # user interrupted
                print("KeyboardInterrupt: User interrupted main program execution")
                break
            except SystemError:
                print("Fatal: SystemError")
                break
            except SystemExit:
                print("Fatal: SystemExit")
                break
        print("Shutting down executor")
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    run_commands()
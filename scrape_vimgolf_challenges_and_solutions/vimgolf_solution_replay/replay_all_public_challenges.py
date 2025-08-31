from library_vim_docker_replay import run_vimgolf_replay
import tempfile
import os
import json
import subprocess
import shutil

challenge_dir = "../challenges"
assert os.path.isdir(
    challenge_dir
), f"Challenge directory {challenge_dir} does not exist"
output_dir = "../replay_output"
print("Using challenge directory:", challenge_dir)
print("Using output directory:", output_dir)
os.makedirs(output_dir, exist_ok=True)

agg_binary_path = "/home/jamesbrown/Downloads/agg-x86_64-unknown-linux-musl"
agg_binary_path = os.path.abspath(agg_binary_path)
assert os.path.isfile(
    agg_binary_path
), f"agg binary path {agg_binary_path} does not exist"
print("Using agg binary path:", agg_binary_path)

challenges = os.listdir(challenge_dir)
for index, challenge_id in enumerate(challenges):
    print("----------------------------------------")
    print("Processing challenge (%s/%s):" % (index + 1, len(challenges)), challenge_id)
    replay_challenge_output_dir = os.path.join(output_dir, challenge_id)
    if os.path.exists(replay_challenge_output_dir):
        print("Output directory already exists, skipping:", replay_challenge_output_dir)
        continue
    challenge_definition_path = os.path.join(
        challenge_dir, challenge_id, "challenge.json"
    )
    if not os.path.isfile(challenge_definition_path):
        print("No challenge definition for challenge:", challenge_id)
        continue
    worst_solution_path = os.path.join(
        challenge_dir, challenge_id, "worst_solution.json"
    )
    if not os.path.isfile(worst_solution_path):
        print("No worst solution for challenge:", challenge_id)
        continue
    with open(challenge_definition_path, "r") as f:
        challenge_definition = json.load(f)
        input_content = challenge_definition["in"]["data"]
        expected_output = challenge_definition["out"]["data"]
    with open(worst_solution_path, "r") as f:
        worst_solution = json.load(f)
        vimgolf_solution = worst_solution["solution"]
    with tempfile.TemporaryDirectory() as tmpdir:
        success = (
            run_vimgolf_replay(
                input_content=input_content,
                expected_output=expected_output,
                vimgolf_solution=vimgolf_solution,
                cast_file_name=os.path.join(tmpdir, "replay.cast"),
                key_action_timestamp_log_file_name=os.path.join(
                    tmpdir, "keylog_timestamps.jsonl"
                ),
            )
            == True
        )
        print("Replay success:", success)
        with open(os.path.join(tmpdir, "success.txt"), "w") as f:
            f.write("success" if success else "failure")
        # create the gif using agg
        subprocess.run(
            [
                agg_binary_path,
                os.path.join(tmpdir, "replay.cast"),
                os.path.join(tmpdir, "replay.gif"),
            ],
            check=True,
            cwd=tmpdir,
        )
        # copy the tmpdir to output_dir
        print("Copying replay output to:", replay_challenge_output_dir)
        copy_success=False
        try:
            shutil.copytree(tmpdir, replay_challenge_output_dir)
            copy_success=True
        finally:
            if not copy_success:
                if os.path.exists(replay_challenge_output_dir):
                    shutil.rmtree(replay_challenge_output_dir)

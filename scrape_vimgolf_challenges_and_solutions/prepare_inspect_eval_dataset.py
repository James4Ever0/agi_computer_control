input_dir = "challenges"
output_file = "vimgolf_public_challenges_inspect_eval.jsonl"

import os
import json

task_ids = os.listdir(input_dir)
task_list = []
for it in task_ids:
    challenge_path = os.path.join(input_dir, it, "challenge.json")
    metadata_path = os.path.join(input_dir, it, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.loads(f.read())
    with open(challenge_path, "r") as f:
        challenge_data = json.loads(f.read())
        _input = challenge_data["in"]["data"]
        _output = challenge_data["out"]["data"]
    task_data = dict(input=_input, target=_output, metadata=metadata, id=it)
    task_list.append(task_data)

with open(output_file, "w") as f:
    for task in task_list:
        f.write(json.dumps(task) + "\n")
print("Output saved to:", output_file)

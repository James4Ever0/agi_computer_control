quizNum = 8

script_path = 'test_open_interpreter_local.sh'
prompt_path = "/tmp/prompt.txt"
 
output_path = "/tmp/openinterpreter_output.log"

import os
from naive_actor_tested import prepare_quiz_by_num

quiz = prepare_quiz_by_num(quizNum)
quiz_prompt = quiz.question

def safe_remove(filepath:str):
    if os.path.exists(filepath):
        print("[+] Removing file:", filepath)
        os.remove(filepath)
    else:
        print('[-] File does not exist:', filepath)

try:
    with open(prompt_path, 'w+', encoding='utf8') as f:
        f.write(quiz_prompt)

    cmd = f"bash -c 'bash {script_path} 2>&1' | tee {output_path}"
    os.system(cmd)

    with open(output_path, 'r', encoding='utf-8') as f:
        output = f.read()

    # evaluate
    success = quiz.evaluate(output)
    if success:
        print(f"[+] Quiz #{quizNum} success")
    else:
        print(f"[-] Quiz #{quizNum} failed")
finally:
    for it in [output_path, prompt_path]:
        safe_remove(it)
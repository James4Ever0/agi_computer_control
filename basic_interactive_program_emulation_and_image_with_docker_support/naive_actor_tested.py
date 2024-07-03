import sys
import os
import time

# you have to run eval in the same environment as the agent, otherwise it will not success.

BENCHMARK_DIR = "../the_frozen_forest_intro/benchmark"
MAX_TRIAL = 4
CMD = "docker run --rm -it alpine:3.7"
INIT_TIME = 3
LOOP_TIME = 1
MAX_EVAL_TRIAL = 10

sys.path.append(BENCHMARK_DIR)

from quiz import Quiz
from naive_actor import NaiveActor, NaiveVocab

def read_and_decode_actor(actor:NaiveActor, errors:str = 'ignore'):
    feedback_bytes = actor.read()
    feedback = feedback_bytes.decode(errors=errors)
    return feedback

# quiz = Quiz(os.path.join(BENCHMARK_DIR, "test_spec/json/test_08.json")) # dig
# quiz = Quiz(os.path.join(BENCHMARK_DIR, "test_spec/json/test_07.json")) # docker python eval
quiz = Quiz(os.path.join(BENCHMARK_DIR, "test_spec/json/test_01.json")) 
actor = NaiveActor(cmd = CMD)
time.sleep(INIT_TIME)

# answers = []
answer = ""
success_items = []

print(f"[*] Quiz:", quiz)

for i in range(MAX_TRIAL):
    print(f"[*] Trial #{i+1}")
    feedback = read_and_decode_actor(actor) # decode into something sensible.
    print(f"[*] Feedback:", repr(feedback))
    it = quiz.evaluate(feedback)
    success_items.append(it)
    # answer += feedback
    # answers.append(feedback)
    # action = NaiveVocab.generate() # random action
    action = "ping bing.com" # the solution
    print(f"[*] Action:", repr(action))
    actor.write(action)
    time.sleep(LOOP_TIME)

feedback = read_and_decode_actor(actor)
it = quiz.evaluate(feedback)
success_items.append(it)

# success_items = sum([int(quiz.evaluate(it)) for it in answers])

# success = quiz.evaluate(answer)

# for i in range(MAX_EVAL_TRIAL):
    # print(f"[*] Eval trial #{i+1}")
    # it = quiz.evaluate(answer)
    # success_items.append(it)
success_count = sum([int(it>0) for it in success_items])
# success_count = sum([int(it) for it in success_items])
print(f"[*] Success rate: {success_count}/{len(success_items)}")
# print(f"[*] Success rate: {success_items}/{total_items}")
# if success:
if success_count != 0:
    print("[+] Quiz Success")
else:
    print("[-] Quiz Failed")

# del actor # to show stats on deletion
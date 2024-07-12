import sys
import os
import time

# you have to run eval in the same environment as the agent, otherwise it will not success.
BENCHMARK_DIR = "../the_frozen_forest_intro/benchmark"
BENCHMARK_JSON_DIR = f"{BENCHMARK_DIR}/test_spec/json/"
DOCKER_ALPINE_CMD = "docker run --rm -it alpine:3.7"

MAX_TRIAL = 4
INIT_TIME = 3
LOOP_TIME = 1
MAX_EVAL_TRIAL = 10

sys.path.append(BENCHMARK_DIR)

from quiz import Quiz
from naive_actor import NaiveActor #, NaiveVocab

class QuizEnv:
    quiz:Quiz
    actor:NaiveActor

def read_and_decode_actor(actor:NaiveActor, errors:str = 'ignore'):
    feedback_bytes = actor.read()
    feedback = feedback_bytes.decode(errors=errors)
    return feedback

def naive_action_generator(feedback:str):
    action = "ping bing.com" 
    return action

def eval_quiz_env(quiz_env, action_generaor):
    actor, quiz= quiz_env.actor, quiz_env.quiz
    ret = eval_action_generator(action_generator, actor, quiz)
    return ret

def eval_action_generator(action_generator, actor, quiz:Quiz, init_time=INIT_TIME, max_trial=MAX_TRIAL, loop_time=LOOP_TIME):
    # quiz = Quiz(os.path.join(BENCHMARK_DIR, "test_spec/json/test_08.json")) # dig
    time.sleep(init_time)

    # answers = []
    # answer = ""
    success_items = []

    print(f"[*] Quiz question:", quiz.question)

    for i in range(max_trial):
        print(f"[*] Trial #{i+1}")
        feedback = read_and_decode_actor(actor) # decode into something sensible.
        print(f"[*] Feedback:", repr(feedback))
        it = quiz.evaluate(feedback)
        success_items.append(it)
        # answer += feedback
        # answers.append(feedback)
        # action = NaiveVocab.generate() # random action
        # action = "ping bing.com" # the solution
        action = action_generator(feedback)
        print(f"[*] Action:", repr(action))
        actor.write(action)
        time.sleep(loop_time)

    feedback = read_and_decode_actor(actor)
    it = quiz.evaluate(feedback)
    success_items.append(it)

    # success_items = sum([int(quiz.evaluate(it)) for it in answers])

    # success = quiz.evaluate(answer)

    # for i in range(MAX_EVAL_TRIAL):
        # print(f"[*] Eval trial #{i+1}")
        # it = quiz.evaluate(answer)
        # success_items.append(it)
    success_count = sum([int(float(it)>0) for it in success_items])
    # success_count = sum([int(it) for it in success_items])
    print(f"[*] Success rate: {success_count}/{len(success_items)}")
    # print(f"[*] Success rate: {success_items}/{total_items}")
    success = success_count != 0
    if success:
    # if success_count != 0:
        print("[+] Quiz Success")
    else:
        print("[-] Quiz Failed")
    return success

def prepare_quiz_and_actor_for_test():
    quiz = Quiz(os.path.join(BENCHMARK_JSON_DIR, "test_07.json")) # docker python eval
    # quiz = Quiz(os.path.join(BENCHMARK_DIR, "test_spec/json/test_01.json")) 
    actor = NaiveActor(cmd = DOCKER_ALPINE_CMD) # actually, shall be called environment, or actor environment
    return quiz, actor

def prepare_quiz_env():
    quiz, actor = prepare_quiz_and_actor_for_test()
    quizEnv = QuizEnv(quiz=quiz, actor=actor)
    return quizEnv

def prepare_and_eval_quiz_env(action_generator):
    quizEnv = prepare_quiz_env()
#    quiz, actor = prepare_quiz_and_actor_for_test()
    eval_quiz_env(quizEnv, action_generator)
#    eval_action_generator(action_generator, actor, quiz)

def main():
    action_generator = naive_action_generator
    prepare_and_eval_quiz_env(action_generator)

if __name__ == "__main__":
    main()

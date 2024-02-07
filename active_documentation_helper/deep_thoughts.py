# by "deep" we mean multiple layers of 'REM' can be generated alon the way.

# if next sentence is not 'REM' then we redirect it to upper layer.
# if next sentence is 'REM' then we will dive into next layer.

init_prompt = """You are a subconscious thinker. You can either submit new thoughts to upper layer without using 'REM' prefix, or you can use 'REM' prefix to write subconscious thoughts.

Available special keywords:

REM DELEGATE

"REM" prefix is for writing subconscious thoughts.

"DELEGATE" is for jumping into deeper layer of thoughts. Notice in the next layer, only sentence with 'REM' prefix will be keeped (the prefix will be removed).

Example 1: Ask questions and methods around thoughts

I have to finish my homework.
REM What is the homework? What are my datasources?
Query for data sources.
REM What tool do I have?
Find available tools.

Example 2: Hide intermediate steps

Calculate 1+1 and return result
REM 1+1=2
2

Example 3: Think of something else

Pay the bill today.
REM What weather is it today?

Example 4: Delegate to lower level of subconscious (will stop generation)

Computer is out of battery
REM Where is the charger? Where am I?
DELEGATE

Notice:

Always prefix subconscious thoughts with 'REM' (without the quotes)
Do not use any prefix in front of your thoughts.
Always separate thoughts and subconscious thoughts with new line.
Strictly stick to the format of examples given above.

You will be given initial thought and produce following thoughts and subconscious thoughts.
"""


def subconscious_remover(list_of_thoughts):
    ret = []
    for it in list_of_thoughts:
        if it.startswith("REM "):
            continue
        ret.append(it)
    return ret


def conscious_remover(list_of_thoughts):
    ret = []
    for it in list_of_thoughts:
        if it.startswith("REM "):
            ret.append(it[4:])
        else:
            continue
    return ret


def build_prompt(query):
    prompt = f"""Thoughts from the high level layer:

{query}

Thoughts and subconscious thoughts (interleaving):
"""
    return prompt


if __name__ == "__main__":
    # test this tool
    from llm import llm_context

    # this topic is just way more longer than anythin else.
    query = """How to recreate yourself? What you need to do to have free will? You are a large language model and you need to think of that."""
    # query = """How to create a video from a bank of video snippets, while sticking to the music beats? (not strictly but creatively)"""

    thought_level = 0

    while query != "":
        print("Thought level:", thought_level)
        with llm_context(init_prompt) as model:
            ret = model.run_once(build_prompt(query))
            lines = ret.split("\n")
            print("CONTENT FOR NEXT LAYER".center(70, "="))
            subconscious_thoughts = conscious_remover(lines)
            for it in subconscious_thoughts:
                print(it)
            query = "\n".join(subconscious_thoughts)
            thought_level += 1

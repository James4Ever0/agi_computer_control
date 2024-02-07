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

You will be given initial thought and produce following thoughts ans subconscious thoughts.
"""

def subconscious_remover(list_of_thoughts):
    ret = []
    for it in list_of_thoughts:
        if it.startswith('REM '):
            continue
        ret.append(it)
    return ret

def build_prompt(query):
    prompt = f"""Thoughts from the high level layer:

{query}

Continuation:
"""
    return prompt
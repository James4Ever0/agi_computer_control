# encountered parsing error -> find other similar keys -> ask the llm if it needs to change -> update keytable

from Levenshtein import ratio

def get_similar_keys(query: str, search_source: list[str]):
    candidates_with_score = [(it, ratio(query, it)) for it in search_source]
    candidates_with_score.sort(key = lambda x: x[1])
    similar_keys_with_score = candidates_with_score[:limit]
    ret = [it[0] for it in similar_keys_with_score]
    return ret

def ask_llm_for_change(invalid_command:str, valid_commands: list[str]):
    prompt = f"""
You have encountered a parsing error during HID script execution.

You submitted an invalid command: {invalid_command}

Similar valid commands: {', '.join(valid_commands)}

Now please select a valid command similar to the invalid command. If no such command exists please do not answer.

Please think step by step.
"""


def update_keytable(keytable:dict[str, str], key: str, new_key: str):
    keytable[newkey] = key

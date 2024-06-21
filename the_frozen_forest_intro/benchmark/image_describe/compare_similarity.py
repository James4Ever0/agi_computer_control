import sys

sys.path.append("../")
from quiz import calculate_semantic_similarity_score

"""
Score:

Similarity between human.txt and agent_identical.txt is 0.8398747444152832
Similarity between human.txt and agent_irrelevant.txt is 0.34165874123573303
Similarity between human.txt and agent_similar.txt is 0.7452027797698975

Conclusion:

Agent will get more rewards if it takes relevant actions.

TODO:

1. Give zero reward for inaction.
2. Use Rerank model instead of sentence embedding.

"""

def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


if __name__ == "__main__":
    answer_file = "human.txt"
    user_answer_file_list = [
        "agent_identical.txt",
        "agent_irrelevant.txt",
        "agent_similar.txt",
    ]

    answer = read_file(answer_file)
    for it in user_answer_file_list:
        user_answer = read_file(it)
        similarity = calculate_semantic_similarity_score(answer, user_answer)
        print("Similarity between {} and {} is {}".format(answer_file, it, similarity))

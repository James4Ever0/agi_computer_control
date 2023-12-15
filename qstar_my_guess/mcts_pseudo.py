import math
import random

# main function for the Monte Carlo Tree Search
def monte_carlo_tree_search(root, time, computational_power):
    while resources_left(time, computational_power):
        leaf = traverse(root)
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)

    return best_child(root)


# function for node traversal
def traverse(node):
    while fully_expanded(node):
        node = best_uct(node)

    # in case no children are present / node is terminal
    return pick_unvisited(node.children) or node


# function for the result of the simulation
def rollout(node):
    while non_terminal(node):
        node = rollout_policy(node)
    return result(node)


# function for randomly selecting a child node
def rollout_policy(node):
    return pick_random(node.children)


# function for backpropagation
def backpropagate(node, result):
    if is_root(node):
        return
    node.stats = update_stats(node, result)
    backpropagate(node.parent, result)  # Pass the result up the tree


# function for selecting the best child
# node with highest number of visits
def best_child(node):
    # Select the child with the highest number of visits
    best_visit_count = -1
    best_child_node = None
    for child in node.children:
        if child.stats.visits > best_visit_count:
            best_visit_count = child.stats.visits
            best_child_node = child
    return best_child_node


# Helper functions
def resources_left(time, computational_power):
    # Check if resources are left
    if time > 0 and computational_power > 0:
        return True
    else:
        return False

def fully_expanded(node):
    # Check if node is fully expanded
    return len(node.children) == node.max_children

def best_uct(node):
    # Select the child node using UCT (Upper Confidence Bound for Trees)
    max_uct = -1
    selected_child = None
    for child in node.children:
        uct_value = calculate_uct(child)
        if uct_value > max_uct:
            max_uct = uct_value
            selected_child = child
    return selected_child

def calculate_uct(node):
    if node.stats.visits == 0:
        return float('inf')
    return (node.stats.wins / node.stats.visits) + math.sqrt(2 * math.log(node.parent.stats.visits) / node.stats.visits)

def pick_unvisited(children):
    # Pick an unvisited child node
    unvisited_children = [child for child in children if child.stats.visits == 0]
    return random.choice(unvisited_children) if unvisited_children else None

def non_terminal(node):
    # Check if node is non-terminal
    return not node.is_terminal

def result(node):
    # Get the result of the simulation
    return node.result

def update_stats(node, result):
    # Update statistics for the node
    node.stats.visits += 1
    node.stats.wins += result  # Assuming result is a win/loss value
    return node.stats

def is_root(node):
    # Check if the node is the root
    return node.parent is None

def pick_random(children):
    # Pick a random child node
    return random.choice(children)

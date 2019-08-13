#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import re
from collections import Counter as counter, defaultdict
from typing import Counter, Dict, List, Set

from tree_accuracy.node import NLGNode


ROOT = "root"
ARG_TEMP_UNIT = "arg_temp_unit"
UNUSED_SCENARIO_ARGS = ["arg_task", "arg_error_reason"]
OPTIONAL_ARGS = ["arg_amount_message", "arg_amount_thread", "arg_bad_arg"]


def is_flat_slot(s: str) -> bool:
    if s and len(s) > 7 and s.lower().startswith("__arg") and s[-2:] == "__":
        return True
    return False


def is_complex_slot(s: str) -> bool:
    if s and len(s) > 8 and s.lower().startswith("[__arg") and s[-2:] == "__":
        return True
    return False


def is_dialog_act(s: str) -> bool:
    s = s.lower()
    if s and (
        (len(s) > 6 and s.startswith("__dg") and s[-2:] == "__")
        or (len(s) > 7 and s.startswith("[__dg") and s[-2:] == "__")
    ):
        return True
    return False


def is_discourse_act(s: str) -> bool:
    s = s.lower()
    if s and (
        (len(s) > 6 and s.startswith("__ds") and s[-2:] == "__")
        or (len(s) > 7 and s.startswith("[__ds") and s[-2:] == "__")
    ):
        return True
    return False


def add_numerical_prefix(s: str, id: int) -> str:
    """
    Add prefix to dialog/discourse acts, e.g.,
    [__dg_inform__ -> [__1_dg_inform__
    """
    return "[__{}_{}_{}__".format(id, s.split("_")[-4], s.split("_")[-3])


def drop_numerical_prefix(s: str) -> str:
    """
    Remove numerical prefix from dialog/discourse acts, e..g.
    [__1_dg_inform__  --> [__dg_inform__]
    """
    # only drop numerical prefixes for dg/ds labels
    return re.sub("_[0-9]+_(d[gs])", "_\g<1>", s)


def add_child(parent, node):
    """
    Add a child node to NLGNode
    """
    for child in parent.children:
        # increment duplicate_id if the node already exists
        if node.equals(child):
            node.duplicate_id = max(node.duplicate_id, child.duplicate_id + 1)
    parent.children.add(node)


def sequence_to_tree(tokens: List[str]) -> NLGNode:
    """
    Convert list of pred/output tokens to a hierarchical tree and return the root node,
    e.g.,
    [__DG_INFORM_2__ supposed to [__ARG_CONDITION_NOT__ not rain ] ]
    =>
    NLGNode("root", children={
        NLGNode("[__DG_INFORM_2__", children={NLGNode("__ARG_CONDITION_NOT__")})
    })
    """
    node = NLGNode(ROOT, None)
    stack = [node]
    counts: Dict[str, int] = defaultdict(int)
    for token in tokens:
        if is_discourse_act(token) or is_dialog_act(token) or is_complex_slot(token):
            node = NLGNode(token, None, duplicate_id=counts[token])
            counts[token] += 1
            stack.append(node)
        elif token == "]" and len(stack) > 1:
            node = stack.pop()
            if len(node.children) == 0 and is_complex_slot(node.label):
                node.label = node.label.replace("[", "")
            add_child(stack[-1], node)
            node = stack[-1]

    while len(stack) > 1:
        node = stack.pop()
        if len(node.children) == 0 and is_complex_slot(node.label):
            node.label = node.label.replace("[", "")
        add_child(stack[-1], node)
    return stack[0] if len(stack) > 0 else node


def scenario_to_tree(tokens: List[str]) -> NLGNode:
    """
    Convert list of scenario tokens to a hierarchical tree and return the root node,
    e.g.,
    [__DG_INFORM__: [__ARG_TASK__: get_forecast ] [__ARG_TEMP_HIGH__: 33 ] ]
    =>
    NLGNode("root", children={
        NLGNode("[__DG_INFORM__:", children={
            NLGNode("[__ARG_TEMP_HIGH__:", children={
                NLGNode("33")
            }),
            NLGNode("[__ARG_TASK__:", children={
                NLGNode("get_forecast")
            })
        })
    })
    """
    node = NLGNode(ROOT, None)
    stack = [node]
    slot_value: List[str] = []
    prefix_idx = 1
    label_counts: Dict[str, int] = defaultdict(int)
    # add dummy token in the beginning and ending
    tokens.insert(0, "")
    tokens.append("")
    for i in range(1, len(tokens) - 1):
        # prev_token, token, next_token = tokens[i - 1], tokens[i], tokens[i + 1]
        token = tokens[i]
        if token.startswith("[__"):
            if is_dialog_act(token) or is_discourse_act(token):
                # dialog or discourse act
                token = add_numerical_prefix(token, prefix_idx)
                prefix_idx += 1
            node = NLGNode(token, None, duplicate_id=label_counts[token])
            label_counts[token] += 1
            stack.append(node)
        elif token == "]":
            if len(slot_value) > 0:
                value_node = NLGNode(" ".join(slot_value), None)
                add_child(node, value_node)
                slot_value = []
            if len(stack) > 1:
                node = stack.pop()
                add_child(stack[-1], node)
                node = stack[-1]
        else:
            slot_value.append(token)

    while len(stack) > 1:
        node = stack.pop()
        add_child(stack[-1], node)
    return stack[0] if len(stack) > 0 else node


def remove_nodes(curr_node: NLGNode) -> None:
    for child in list(curr_node.children):
        for unused_arg in UNUSED_SCENARIO_ARGS:
            if unused_arg in child.label:
                curr_node.children.remove(child)
        else:
            remove_nodes(child)


def get_flatten_nodes(
    curr_node: NLGNode, is_scenario: bool, prefix: str = ""
) -> Counter[NLGNode]:
    """
    Flatten a hierarchical tree and return a set of leaf node. The label of
    a leaf node is a concatenation of the path to the leaf node.
    """
    if is_scenario and curr_node.label == ROOT:
        remove_nodes(curr_node)
    nodes: Counter[NLGNode] = counter()
    label = curr_node.label.lower()
    if is_complex_slot(label):
        label = label[1:]
    new_prefix = label if len(prefix) == 0 else "{},{}".format(prefix, label)
    if len(curr_node.children) == 0:
        nodes[NLGNode(new_prefix, curr_node.span)] += 1
    for child in curr_node.children:
        children_nodes = get_flatten_nodes(child, is_scenario, new_prefix)
        nodes.update(children_nodes)
    return nodes


def prune_scenario_tree(
    scenario_tree: NLGNode, scenario_nodes: Counter[NLGNode]
) -> None:
    """
    Prune scenario tree by removing aggregation nodes, e.g.,
    NLGNode("root", children={
        NLGNode("__1_dg_inform__", children=NLGNode("__arg_location__")),
        NLGNode("__2_dg_inform__", children=NLGNode("__arg_location__")),
    }),
    Counter[
        NLGNode("root,[__1_dg_inform__,__arg_location__,menlo park")
        NLGNode("root,[__2_dg_inform__,__arg_location__,menlo park"),
    ]
    -> NLGNode("root", children={
        NLGNode("__1_dg_inform__", children=NLGNode("__arg_location__")),
    })
    """

    def remove_child(curr_node, label_list):
        if curr_node is None or len(label_list) == 0:
            return
        label = drop_numerical_prefix(label_list[0])
        if is_dialog_act(label) or is_discourse_act(label):
            next_node = None
            for child in curr_node.children:
                if child.label == label_list[0]:
                    next_node = child
                    break
            remove_child(next_node, label_list[1:])
        elif is_complex_slot(label) or is_flat_slot(label):
            for child in curr_node.children:
                if child.label.replace("[", "") == label_list[0]:
                    curr_node.children.remove(child)
                    return

    # sort node labels by prefix
    sorted_labels = [node.label for node in scenario_nodes]
    sorted_labels.sort()
    label_set: Set[str] = set()
    for label in sorted_labels:
        slot_label = get_slot_label(label)
        if slot_label == ROOT or slot_label not in label_set or ARG_TEMP_UNIT in label:
            label_set.add(slot_label)
        else:
            remove_child(scenario_tree, label.split(",")[1:])
    # Make a deep copy of scenario tree. This step is necessary to refresh the
    # hash value of internal nodes in later comparison
    return copy_scenario_tree(scenario_tree)


def copy_scenario_tree(curr_node):
    """
    Make a deep copy of scenario tree and truncate leaf value nodes.
    """
    new_node = NLGNode(
        drop_numerical_prefix(curr_node.label),
        curr_node.span,
        None,
        curr_node.duplicate_id,
    )
    is_dg = is_dialog_act(new_node.label) or is_discourse_act(new_node.label)
    new_children = []
    for child in list(curr_node.children):
        # remove leaf argument node
        if len(child.children) > 0 or child.label.startswith("[__"):
            new_children.append(copy_scenario_tree(child))
    for child in new_children:
        add_child(new_node, child)
    if len(new_node.children) == 0 and not is_dg:
        new_node.label = new_node.label.replace("[", "")
    return new_node


def get_slot_label(label):
    """
    Get slot parts from label of a flatten node, e.g.,
    "root,[__1_dg_inform__,__arg_temp__,33" -> "root __arg_temp__ 33"
    """
    slot_label = ""
    for s in label.split(","):
        if not s.startswith("[__"):
            slot_label += s + " "
    return slot_label.strip()


def get_filtered_children(node: NLGNode) -> Set[NLGNode]:
    """
    Filters out all children of a node that are unused and returns the new set
    """
    new_children = set()
    for n in node.children:
        if not any(k in n.label for k in UNUSED_SCENARIO_ARGS):
            new_children.add(n)
    return new_children


# Anything that's not a DG/DS/ARG is a raw value
def is_raw_value(s: str) -> bool:
    s = s.lower()
    return "_dg_" not in s and "_ds_" not in s and "_arg_" not in s


def is_leaf_node(node: NLGNode) -> bool:
    children = get_filtered_children(node)
    if len(children) > 1:
        #         print("more than 1 child")
        return False
    if len(children) == 0:
        return True
    child = list(children)[0]
    return is_raw_value(child.label)


def is_optional_arg(label: str) -> bool:
    return any(k in label for k in OPTIONAL_ARGS)


def find_aggregation_nodes(
    tree: NLGNode, values: Dict[str, Set[NLGNode]], dupes: Set[str]
):
    """
    Given a scenario tree, finds all nodes that could potentially be aggregated,
    based on overlap between leaf node values.
    NB: This ignores the identity of the nodes themselves, so any two nodes
    could be marked as overlapping regardless of their labels (e.g. location and
    datetime could be aggregated if, for some reason, they shared a value.)
    """
    children = get_filtered_children(tree)
    if is_leaf_node(tree) and len(children) > 0:
        child = list(children)[0]
        if child.label not in values:
            values[child.label] = set()
        else:
            dupes.add(child.label)
        values[child.label].add(tree)
        return
    for child in get_filtered_children(tree):
        find_aggregation_nodes(child, values, dupes)


def get_aggregation_nodes(tree: NLGNode) -> Dict[str, Set[NLGNode]]:
    values: Dict[str, Set[NLGNode]] = {}
    dupes: Set[str] = set()
    find_aggregation_nodes(tree, values, dupes)
    # Only treat nodes with repeated values as aggregated
    values = {k: v for (k, v) in values.items() if k in dupes}
    return values


def get_flattened_aggregation_nodes(nodes: Dict[str, Set[NLGNode]]) -> Set[NLGNode]:
    all_agg_nodes = set()
    for v in nodes.values():
        all_agg_nodes.update(v)
    return all_agg_nodes


def remove_matches(
    pred_matched: NLGNode,
    scenario_matched: NLGNode,
    matches: Dict[NLGNode, Set[NLGNode]],
) -> bool:
    """
    Given a prediction node and the scenario node that it was matched to,
    recursively removes match links between any children of the two nodes.
    """
    for pred_child in pred_matched.children:
        if pred_child not in matches or len(matches[pred_child]) == 0:
            continue
        removed = True
        for scenario_child in scenario_matched.children:
            if scenario_child in matches[pred_child]:
                matches[pred_child].remove(scenario_child)
            removed = removed and remove_matches(pred_child, scenario_child, matches)
        if not removed or len(matches[pred_child]) == 0:
            return False
    return True


def prune_matches(matched_node: NLGNode, matches: Dict[NLGNode, Set[NLGNode]]) -> bool:
    """
    Given a scenario node, removes it from the hypotheses list for any node that
    was matched to it. Also recursively removes children of the given scenario
    node from the hypothesis list (see remove_matches).
    """
    if ARG_TEMP_UNIT in preprocess_label(matched_node.label):
        return True
    for (node, potential_matches) in matches.items():
        removed = True
        if matched_node in potential_matches:
            potential_matches.remove(matched_node)
            removed = remove_matches(node, matched_node, matches)
        if len(potential_matches) == 0 or not removed:
            return False
    return True


def check_scenario_completeness(
    scenario_nodes: Set[NLGNode],
    matches: Dict[NLGNode, Set[NLGNode]],
    aggregation_nodes: Set[NLGNode],
) -> bool:
    """
    Checks whether all the given nodes of the scenario have been matched.
    Accounts for aggregation and optional args.
    """
    mn = set()
    complete = True
    for v in matches.values():
        mn.update(v)
    for node in scenario_nodes:
        if is_leaf_node(node):
            # Every leaf node must be matched, or be aggregable or optional
            if (
                node not in mn
                and node not in aggregation_nodes
                and not is_optional_arg(node.label)
            ):
                complete = False

        else:
            # if a non-terminal node has no matches, all its children must be
            # aggregable or optional
            if not all(
                c in aggregation_nodes or c in mn or is_optional_arg(c.label)
                for c in get_filtered_children(node)
            ):
                complete = False
            else:
                # If all the non-terminals children have been matched, ensure
                # that the non-terminal is marked as matched as well
                matches[node] = {node}
    return complete


def compare_trees_impl(
    scenario_nodes: List[NLGNode],
    pred_tree: NLGNode,
    aggregation_nodes: Dict[str, Set[NLGNode]],
    matches: Dict[NLGNode, Set[NLGNode]],
) -> bool:
    """
    Tries to match the given prediction node (pred_tree) to all the given
    scenario_nodes, and checks whether there's at least one match. Each node
    recursively checks its children.

    `matches` is used to keep track of all current match hypotheses (it's a map
    from the prediction node to all possible scenario nodes that it could be
    matched to.)

    aggregation_nodes is a map containing all nodes that could be aggregated
    together since they share the same value. (map from the aggregated value to
    a set of all nodes that share it.)
    """
    all_agg_nodes = get_flattened_aggregation_nodes(aggregation_nodes)
    pred = preprocess_label(pred_tree.label)
    matches_found: Set[NLGNode] = set()
    for node in scenario_nodes:
        label = preprocess_label(drop_numerical_prefix(node.label))
        children = get_filtered_children(node)
        if pred != label:
            continue
        new_matches: Dict[NLGNode, Set[NLGNode]] = {}
        found = False
        if len(pred_tree.children) == 0 and len(children) == 0:
            # If the prediction and the scenario node have no children, match
            found = True
        elif len(pred_tree.children) == 0 and is_leaf_node(node):
            # If the prediction node has no children and the scenario node
            # is a leaf node, this is a match
            found = True
        else:
            # Otherwise, check if every child of the prediction node has a valid
            # match in this subtree of the scenario
            found = True
            for child in pred_tree.children:
                found = found and compare_trees_impl(
                    list(children), child, aggregation_nodes, new_matches
                )
            # also ensure that every node of the scenario is matched
            found = found and check_scenario_completeness(
                children, new_matches, all_agg_nodes
            )

        if found:
            # update current hypotheses in match with new matches found
            matches_found.add(node)
            for (k, v) in new_matches.items():
                if k not in matches:
                    matches[k] = v
                else:
                    matches[k].update(v)

    if len(matches_found) == 0:
        return False
    if len(matches_found) == 1:
        # If exactly one match was found for this node, we need to invalidate
        # matches for any other nodes that have matched the same part of the
        # scenario.
        # NB: If this node has multiple matches, we don't do any kind of
        # tie-breaking here.. theoretically it's possible that each of the
        # matches for this node could overlap with some other nodes that
        # each have exactly one match (e.g. {1: [A,B], 2: [A], 3: [B]}),
        # but this seems like an edge case
        pruned = prune_matches(list(matches_found)[0], matches)
        if not pruned:
            return False

    matches[pred_tree] = matches_found
    return True


def preprocess_label(label: str) -> str:
    return label.replace("[", "").lower()


def compare_trees(scenario_tree: NLGNode, pred_tree: NLGNode) -> bool:
    """
    Compares the scenario tree against the predicted response tree. Returns true
    if the predicted response matches the scenario, and false otherwise.
    """
    if scenario_tree == pred_tree:
        return True
    pred = preprocess_label(pred_tree.label)
    assert pred == ROOT, "compare_trees should be called with root nodes"

    # find all nodes that could be aggregated together
    aggregation_nodes = get_aggregation_nodes(scenario_tree)
    matches: Dict[NLGNode, Set[NLGNode]] = {}
    for child in pred_tree.children:
        # Check whether each child of the root prediction tree has at least
        # one match in the scenario (each child is checked recursively in
        # compare_trees_impl)
        m = compare_trees_impl(
            scenario_tree.children, child, aggregation_nodes, matches
        )
        if not m:
            # False positives
            return False

    matched_nodes: Set[NLGNode] = set()
    for v in matches.values():
        matched_nodes.update(v)
    # Check if any of the children of the root scenario node haven't been matched
    # (the recursive check is performed in compare_trees_impl)
    for child in scenario_tree.children:
        if child not in matched_nodes:
            return False

    # For each group of aggregated nodes, check that at least one was matched
    # in the predicted response
    for (_, group) in aggregation_nodes.items():
        if not any(k in matched_nodes for k in group):
            return False
    return True

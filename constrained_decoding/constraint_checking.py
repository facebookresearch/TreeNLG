#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from collections import defaultdict
from typing import Dict, List, Set


OPEN_BRACKET = "["
CLOSE_BRACKET = "]"
IGNORE_NON_TERMINALS = {
    "__ARG_TASK__",
    "__ARG_BAD_ARG__",
    "__ARG_ERROR_REASON__",
    "__ARG_TEMP_UNIT__",
}
IGNORE_NODE_REGEX = r"\[{} [a-z_]+ \]"


def split_and_strip(s):
    """
    Split string by space and strip any whitespace
    """
    return [tok.strip() for tok in s.split(" ") if tok.strip()]


class VocabMeta:
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"


class DecodingState:
    """
    Multiple possible states exist during the decoding due to aggregation and
    interchangeable order of children.
    """

    def __init__(
        self,
        parent: int,
        coverage: Set[int],
        cmplt_coverage: Set[int],
        agg_coverage: Set[int],
        node_alignment: Dict[int, int],
    ):
        self.parent: int = parent
        # node ids that have been encountered
        self.coverage: Set[int] = coverage
        # node ids that their descendants and themselves have been encountered
        self.cmplt_coverage: Set[int] = cmplt_coverage
        # tracks nodes that provide aggregration and have occured or
        # can still occur in the future. If a node gets aggregated (i.e. we
        # assume that it will be covered by another node) then it is removed
        # from self.agg_coverage
        self.agg_coverage: Set[int] = agg_coverage
        # whether this state has met all constraints
        self.finished: bool = False
        # map of input node id - > idx of non-terminal token in output
        self.node_alignment: Dict[int, int] = node_alignment

    def __str__(self):
        return "{} {} {} {} {}".format(
            self.parent,
            str(self.coverage),
            str(self.cmplt_coverage),
            str(self.agg_coverage),
            str(self.node_alignment),
        )

    def __repr__(self):
        return str(self)

    def can_aggregate(
        self, children: Set[int],
        coverage_options: Dict[int, Set[int]],
        children_map: Dict[int, Set[int]]
    ) -> bool:
        """
        Return True if current non-terminal can consume a closing brace
        taking possible aggregation of missing children into account
        args:
            children: expected children
            coverage_options: nodes that can cover a missing node
        """
        missing_children = children - self.coverage
        stack = list(missing_children)
        while stack:
            curr_node = stack.pop(-1)
            curr_opts = coverage_options.get(curr_node, {curr_node})
            if len(curr_opts & self.cmplt_coverage) > 0:
                continue
            elif len(curr_opts & self.agg_coverage) > 1:
                stack.extend(children_map[curr_node])
            else:
                return False
        return True


class TreeConstraints:
    """
    Build and check constraints during constrained decoding. Each TreeConstraints
    object corresponds to one candidate in beam.
    """

    def __init__(
        self,
        input_tree: str,
        order_constr: bool = False
    ):
        # activate order constraints
        self.order_constr = order_constr
        # map from non-terminal tokens to input nodes at which they're valid.
        self.non_terminal_map: Dict[str, Set[int]] = defaultdict(set)
        # map input nodes to their parents
        self.parent_map: Dict[int, int] = {}
        # map input nodes to their children
        self.children_map: Dict[int, Set[int]] = defaultdict(set)
        # map from terminal nodes to their arg values, if any
        self.terminal_map: Dict[int, str] = {}
        # map from node id to non-terminal it corresponds to
        self.node_map: Dict[int, str] = {}
        # map from node to other nodes that can cover it through aggregation
        self.coverage_options: Dict[int, Set[int]] = {}
        # total number of non-terminals expected in output
        self.total_non_terminals: int = 0
        self.satisfied: bool = False

        self.valid_input = self.parse_input(self.preprocess_input(input_tree))
        self.parse_aggregation_rules()

        agg_coverage: Set[int] = set()
        for agg_options in self.coverage_options.values():
            agg_coverage = agg_coverage | agg_options
        # current list of possible states
        self.states: List[DecodingState] = [DecodingState(-1, set(), set(), agg_coverage, {})]
        # if we're currently consuming a non-terminal whose constraint is to be ignored
        self.ignoring_non_terminal = 0
        # debug
        self.tokens: List[str] = []
        self.input_tree = self.preprocess_input(input_tree)

    def _parse_non_terminal(self, name: str, node_id: int, parent_id: int):
        self.non_terminal_map[name].add(node_id)
        self.node_map[node_id] = name
        self.parent_map[node_id] = parent_id
        self.children_map[parent_id].add(node_id)
        self.total_non_terminals += 1

    def preprocess_input(self, tree: str) -> str:
        # remove task nodes like (__arg_task__ my_task )
        tree = tree.replace(VocabMeta.EOS_TOKEN, "").replace(VocabMeta.PAD_TOKEN, "")
        for non_term in IGNORE_NON_TERMINALS:
            for match in re.findall(IGNORE_NODE_REGEX.format(non_term), tree):
                tree = tree.replace(match, "")
        return tree

    def parse_input(self, tree: str) -> bool:
        """
        Populate non_terminal_map, parent_map, children_map and verify
        that input is a valid tree
        """
        try:
            tree = tree.strip()
            # non terminal should not occur before opening brace
            assert tree[0] == OPEN_BRACKET, "Could not parse \n" + tree
            nt_start = -1  # start pos of first un-parsed non-terminal in tree string
            # -1 is dummy parent node that will be parent of all root nodes
            # in case of multiple trees
            curr_parent = -1
            node_id = -1
            self.parent_map[0] = -1
            allowed_close_brackets = 0

            def non_terminal_found(nt_start: int, i: int):
                # process text found between two braces
                # eg: "(NON_TERMINAL )" or "(NON_TERMINAL ("
                last_nt = tree[nt_start:i].strip()
                if " " in last_nt:
                    # has an arg val, eg: (NON_TERMINAL arg value)
                    toks = [tok for tok in last_nt.split(" ") if tok.strip()]
                    assert len(toks) > 1
                    last_nt, arg_val = toks[0], " ".join(toks[1:])
                    self.terminal_map[node_id] = arg_val
                self._parse_non_terminal(last_nt, node_id, curr_parent)

            for i in range(len(tree)):
                # parse token between nt_start and occurence of open/close brace
                if tree[i] == OPEN_BRACKET:
                    allowed_close_brackets += 1
                    if nt_start != -1:
                        non_terminal_found(nt_start, i)
                        curr_parent = node_id
                    node_id += 1
                    # open brace represents start of next token
                    nt_start = i + 1
                elif tree[i] == CLOSE_BRACKET:
                    allowed_close_brackets -= 1
                    assert allowed_close_brackets >= 0, "Could not parse \n" + tree
                    if nt_start != -1:
                        non_terminal_found(nt_start, i)
                    else:
                        curr_parent = self.parent_map[curr_parent]
                    # we haven't encountered start of next token yet,
                    # set nt_start to invalid
                    nt_start = -1
            assert curr_parent == -1, "Could not parse \n" + tree
            return True
        except Exception as e:
            print("Couldn't parse: ")
            print(tree)
            print(e)
            return False

    def _subtrees_equal(self, t1: int, t2: int) -> bool:
        """
        Check if subtrees t1 and t2 are equal
        """
        if self.node_map[t1] != self.node_map[t2]:
            return False
        if t1 in self.children_map and t2 in self.children_map:
            # both have children
            children1 = sorted(list(self.children_map[t1]))
            children2 = sorted(list(self.children_map[t2]))
            if len(children1) != len(children2):
                return False
            for child1, child2 in zip(children1, children2):
                if not self._subtrees_equal(child1, child2):
                    return False
            return True
        elif t1 in self.terminal_map and t2 in self.terminal_map:
            # leaf node with arg value
            return self.terminal_map[t1] == self.terminal_map[t2]
        elif t1 not in self.terminal_map and t2 not in self.terminal_map:
            # leaf node without arg value
            return True
        else:
            return False

    def parse_aggregation_rules(self):
        """
        Identify and store nodes that can be aggregated
        This populates self.coverage_options with nodes in the tree
        who have identical "cousins"
        """
        stack = list(self.children_map[-1])  # all root notes in input forest
        while stack:
            curr_node = stack.pop(-1)
            if curr_node in self.coverage_options:
                # this node's cousin has already done the work
                continue
            for candidate in range(self.total_non_terminals - 1, -1, -1):
                if (
                    candidate != curr_node
                    and candidate not in self.coverage_options
                    and self._subtrees_equal(curr_node, candidate)
                ):
                    cov = self.coverage_options.get(curr_node, set())
                    cov.add(curr_node)
                    cov.add(candidate)
                    self.coverage_options[curr_node] = cov
                    self.coverage_options[candidate] = cov
            stack.extend(self.children_map[curr_node])

    def meets_all(self) -> bool:
        return self.satisfied

    def _invalid(self) -> bool:
        self.satisfied = False
        self.states = []
        return False

    def _accept_non_terminal(self, nt: str, idx: int) -> bool:
        """
        Try and accept the given non-terminal
            nt: non terminal token
            idx: index of token in output
        """
        new_states = []
        for state in self.states:
            # check each state if it accepts the nt
            uncovered_children = [
                node for node in self.children_map[state.parent]
                if node not in state.coverage
            ]
            if (
                self.order_constr and
                state.parent != -1 and
                self.node_map[state.parent] == '__DS_JOIN__'
            ):
                uncovered_children.sort()
                uncovered_children = uncovered_children[:1]
            for node in self.non_terminal_map[nt]:
                # do for all possible nodes the nt can map to
                if node not in uncovered_children:
                    # this state can't accept this node
                    continue
                alignment = dict(state.node_alignment)
                # track alignment between input node id and output token idx
                alignment[node] = idx
                new_states.append(
                    DecodingState(
                        node,
                        set(state.coverage) | {node},
                        set(state.cmplt_coverage),
                        set(state.agg_coverage),
                        alignment,
                    )
                )
        self.states = new_states
        return self._invalid() if len(new_states) == 0 else True

    def _remove_coverage_options(self, state: DecodingState):
        missing_children = self.children_map[state.parent] - state.coverage

        def remove_subtrees(missing_children):
            for child in missing_children:
                if child in state.agg_coverage:
                    state.agg_coverage.remove(child)
                remove_subtrees(self.children_map[child])

        remove_subtrees(missing_children)

    def _does_cmplt_cover(self, state) -> bool:
        stack = [state.parent]
        while stack:
            curr_node = stack.pop(-1)
            curr_opts = self.coverage_options.get(curr_node, {curr_node})
            if len(curr_opts & state.cmplt_coverage) > 0:
                continue
            elif len(curr_opts & state.coverage) > 0:
                stack.extend(self.children_map[curr_node])
            else:
                return False
        return True

    def _accept_closing_brace(self) -> bool:
        new_states = []
        is_complete = False
        for state in self.states:
            # update each state with closing brace, save valid states for next step
            def close_node():
                if self._does_cmplt_cover(state):
                    state.cmplt_coverage.add(state.parent)
                state.parent = self.parent_map[state.parent]
                new_states.append(state)
                covered_nodes = set(state.coverage)
                for node in state.agg_coverage:
                    covered_nodes = covered_nodes | self.coverage_options[node]
                # check if state has met all constraints
                if (
                    state.parent == -1
                    and len(covered_nodes) == self.total_non_terminals
                ):
                    self.satisfied = True
                    return True

            if state.parent == -1:
                # we've already closed all open nodes
                continue
            elif (
                # node w/o children can always consume closing brace
                state.parent not in self.children_map
                # all children of node have been covered
                or self.children_map[state.parent] <= state.coverage
            ):
                is_complete = close_node()
            elif state.can_aggregate(
                self.children_map[state.parent],
                self.coverage_options,
                self.children_map
            ):
                self._remove_coverage_options(state)
                is_complete = close_node()
        if is_complete:
            # only one finished state should exist at the end of checking
            self.finished_state = state
        if not is_complete and not new_states:
            return self._invalid()
        else:
            self.states = new_states
            return True

    def next_token(self, token: str, idx: int) -> bool:
        """
        Updates states based on next token. Returns True if token was
        consumed successfully.
        Args:
            token: next token in string to check constraints for
            idx: index of token in string
        """
        self.tokens.append(token)
        try:
            if not self.valid_input:
                return self._invalid()
            if token == VocabMeta.EOS_TOKEN or token == VocabMeta.PAD_TOKEN:
                return True
            if self.satisfied and (token[0] == OPEN_BRACKET or token == CLOSE_BRACKET):
                # can't accept another token after tree is complete
                return self._invalid()
            if not self.states:
                # there are no possible states left
                return False
            if token[0] == OPEN_BRACKET and token[1:] in IGNORE_NON_TERMINALS:
                self.ignoring_non_terminal += 1
                return True
            elif token[0] == OPEN_BRACKET:
                return self._accept_non_terminal(token[1:], idx)
            elif token == CLOSE_BRACKET and self.ignoring_non_terminal > 0:
                self.ignoring_non_terminal -= 1
                return True
            elif token == CLOSE_BRACKET:
                return self._accept_closing_brace()
            else:
                # token that's not non-terminal or closing brace
                return True
        except Exception as e:
            print("Failed at: ", self.tokens)
            print(self.input_tree)
            raise e

    def parse(self, target: str) -> bool:
        """
        Returns True if the given str successfully meets constraints
        """
        for i, token in enumerate(split_and_strip(target)):
            if not self.next_token(token, i):
                return False
        return self.meets_all()

    def nominate_nt(self) -> Set[int]:
        """
        Nominate possible non-terminals for next step beam search
        """
        assert self.valid_input
        nominated_nt = set()
        if not self.states:
            return nominated_nt
        if self.satisfied:
            nominated_nt.add(VocabMeta.EOS_TOKEN)
            return nominated_nt
        nominated_nt.update(IGNORE_NON_TERMINALS)
        if self.ignoring_non_terminal > 0:
            nominated_nt.add(CLOSE_BRACKET)
        for state in self.states:
            uncovered_children = [
                node for node in self.children_map[state.parent]
                if node not in state.coverage
            ]
            if (
                self.order_constr and
                state.parent != -1 and
                self.node_map[state.parent] == '__DS_JOIN__'
            ):
                uncovered_children.sort()
                uncovered_children = uncovered_children[:1]
            nominated_nt.update([self.node_map[nt] for nt in uncovered_children])
            if state.parent == -1:
                continue
            elif (state.parent not in self.children_map or
                self.children_map[state.parent] <= state.coverage or
                state.can_aggregate(
                    self.children_map[state.parent],
                    self.coverage_options,
                    self.children_map
                )
            ):
                nominated_nt.add(CLOSE_BRACKET)
        return nominated_nt

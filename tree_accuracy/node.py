#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import AbstractSet, Any, NamedTuple, Optional, Set


class Span(NamedTuple):
    """
    Span of a node in an intent-slot tree.

    Attributes:
        start: Start position of the node.
        end: End position of the node (exclusive).
    """

    start: int
    end: int


class Node:
    """
    Node in an intent-slot tree, representing either an intent or a slot.

    Attributes:
        label (str): Label of the node.
        span (Span): Span of the node.
        children (:obj:`set` of :obj:`Node`): Children of the node.
    """

    __slots__ = "label", "span", "children"

    def __init__(
        self, label: str, span: Span, children: Optional[AbstractSet["Node"]] = None
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "span", span)
        object.__setattr__(
            self, "children", children if children is not None else set()
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.label == other.label  # noqa
            and self.span == other.span  # noqa
            and self.children == other.children  # noqa
        )

    def get_depth(self) -> int:
        return 1 + max(
            (child.get_depth() for child in self.children), default=0  # noqa
        )


class NLGNode(Node):
    def __init__(
        self,
        label: str,
        span: Span,
        children: Set["NLGNode"] = None,
        duplicate_id: int = 0,
    ) -> None:
        super().__init__(label, span, children)
        self.duplicate_id = duplicate_id

    def __hash__(self):
        return hash(
            (self.label, self.span, frozenset(self.children), self.duplicate_id)
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NLGNode):
            return NotImplemented
        return (
            self.label == other.label
            and self.span == other.span
            and self.children == other.children
            and self.duplicate_id == other.duplicate_id
        )

    # unlike __eq__ operator, equals only check for whether two nodes have same content.
    def equals(self, other) -> bool:
        return (
            self.label == other.label
            and self.span == other.span
            and self.children == other.children
        )

    def print_node(self) -> None:
        print(self.label, self.duplicate_id, self.get_depth(), self.__hash__())
        for child in self.children:
            child.print_node()

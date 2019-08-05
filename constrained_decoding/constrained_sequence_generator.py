#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from copy import deepcopy

import torch
from pytorch_translate import utils as pytorch_translate_utils

from constrained_decoding.constraint_checking import TreeConstraints
from constrained_decoding.sequence_generator import SequenceGenerator


# non-terminal prefix
NT_PREFIX = "b__"


def bracketize(s):
    """
    Change the prefix of non-terminal tokens b__ to [__, i.e.,
    b__dg_inform__ to [__dg_inform__.
    """
    tokens = s.split()
    if len(tokens) <= 1:
        return re.sub(r"^%s" % NT_PREFIX, "[__", s)
    else:
        return " ".join([bracketize(t) for t in tokens])


class NLGFairseqSequenceGenerator(SequenceGenerator):
    def __init__(self, models, src_dict, tgt_dict, config):
        super().__init__(models, tgt_dict, **config._asdict())
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def generate_hypo(self, repacked_inputs, maxlen_a=0.0, maxlen_b=None):
        if maxlen_b is None:
            maxlen_b = self.maxlen
        src_tokens = repacked_inputs["src_tokens"]
        srclen = pytorch_translate_utils.get_source_tokens_tensor(src_tokens).size(1)
        hypos = self.generate(
            repacked_inputs,
            beam_size=self.beam_size,
            maxlen=int(maxlen_a * srclen + maxlen_b),
            # If we need to generate predictions with teacher forcing, this
            # won't work. Right now this is fine.
            prefix_tokens=None,
        )
        return self._pick_hypothesis_unpack_output(hypos)

    @staticmethod
    def _pack_input_for_fairseq(src_tokens, src_lengths):
        return {"src_tokens": src_tokens, "src_lengths": src_lengths}

    @staticmethod
    def _pick_hypothesis_unpack_output(all_hypos):
        """
        For now, we just pick the first hypothesis returned by fairseq and we
        return just the "tokens" as output
        """
        results = []
        for hypo in all_hypos:
            beam_results = []
            for prediction in hypo:
                beam_results.append(prediction["tokens"])
            results.append(beam_results)
        return results

    def _build_constraints(self, src_tokens, beam_size):
        """
        Returns list of constraint objects of size (bsz * beam_size, )
        """
        srcs = [" ".join([self.src_dict[tok] for tok in row]) for row in src_tokens]
        srcs = [s.replace(self.tgt_dict[self.tgt_dict.bos()], "") for s in srcs]
        srcs = [s.replace(self.tgt_dict[self.tgt_dict.eos()], "") for s in srcs]
        constraints = [TreeConstraints(bracketize(t)) for t in srcs]
        bbeam_constraints = []
        for constraint in constraints:
            bbeam_constraints.extend([deepcopy(constraint) for i in range(beam_size)])
        self.constraint_penalty = [0.0] * len(bbeam_constraints)
        return bbeam_constraints

    def _apply_constraint_penalty(self, scores):
        """
        Penalize unmet constraints
        """
        assert len(self.constraint_penalty) == scores.size(0)
        scores += torch.tensor(self.constraint_penalty, device=scores.device).unsqueeze(
            1
        )

    def _update_constraints(self, constraints, next_tokens, idx):
        """
        Based on tokens consumed, update constraints and penalties for next step
        """
        assert len(constraints) == len(next_tokens)
        self.constraint_penalty = [
            0.0
            if constraint.next_token(bracketize(self.tgt_dict[token]), idx)
            else float("-Inf")
            for constraint, token in zip(constraints, next_tokens)
        ]

    def _reorder_constraints(self, constraints, new_indices):
        """
        Equivalent to constraints[new_indices] if both were Tensors.
        """
        # deepcopy is needed since the same candidate can appear in
        # multiple locations
        return [deepcopy(constraints[idx]) for idx in new_indices]

    def _apply_eos_constraints(self, constraints, eos_bbsz_idx, eos_scores):
        """
        Only allow EOS for candidates that satisfy all constraints
        Returns filters eos indices and scores
        """
        eos_constraints = self._reorder_constraints(constraints, eos_bbsz_idx)
        meets_constraints = []
        for i, con in enumerate(eos_constraints):
            if con.meets_all():
                meets_constraints.append(i)
        meets_constraints = torch.tensor(
            meets_constraints, device=eos_bbsz_idx.device, dtype=torch.long
        )
        return eos_bbsz_idx[meets_constraints], eos_scores[meets_constraints]

    def _finalize_constrained_results(self, finalized, device):
        """
        Deal with potentially empty results after beam search
        """
        for item in finalized:
            if len(item) == 0:
                item.append(
                    {
                        "tokens": torch.LongTensor([self.eos], device=device),
                        "score": -float("-Inf"),
                    }
                )

from fairseq import search

from fairseq.tasks import FairseqTask


def build_generator(self, args):
    search_strategy = search.BeamSearch(self.target_dictionary)
    from ..constrained_decoding import ConstrainedSequenceGenerator
    seq_gen_cls = ConstrainedSequenceGenerator
    dicts = (self.source_dictionary, self.target_dictionary)
    return seq_gen_cls(
        *dicts,
        beam_size=getattr(args, 'beam', 5),
        max_len_a=getattr(args, 'max_len_a', 0),
        max_len_b=getattr(args, 'max_len_b', 200),
        min_len=getattr(args, 'min_len', 1),
        normalize_scores=(not getattr(args, 'unnormalized', False)),
        len_penalty=getattr(args, 'lenpen', 1),
        unk_penalty=getattr(args, 'unkpen', 0),
        temperature=getattr(args, 'temperature', 1.),
        match_source_len=getattr(args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        search_strategy=search_strategy,
    )

FairseqTask.build_generator = build_generator

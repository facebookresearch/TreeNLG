from fairseq import options
from fairseq.options import add_generation_args


def add_constrainted_generation_args(parser):
    group = add_generation_args(parser)
    group.add_argument('--order-constr', action='store_true',
                       help='activate order constraint')
    return group

options.add_generation_args = add_constrainted_generation_args

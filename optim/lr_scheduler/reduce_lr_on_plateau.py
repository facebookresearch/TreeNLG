import torch.optim.lr_scheduler

from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateau

def __init__(self, args, optimizer):
    super(self.__class__, self).__init__(args, optimizer)
    if len(args.lr) > 1:
        raise ValueError(
            'Cannot use a fixed learning rate schedule with reduce_lr_on_plateau.'
            ' Consider --lr-scheduler=fixed instead.'
        )
    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer.optimizer, factor=args.lr_shrink, patience=args.lr_patience,
        threshold=args.lr_threshold)
    warmup_end_lr = args.lr[0]
    """if no warm up, sets initial lr to be args.lr[0]"""
    if args.warmup_init_lr < 0:
        args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

    """ linearly warmup for the first args.warmup_updates"""
    if args.warmup_updates > 0:
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
    """ this flag is either set from arg when no warm up, or set by step_update() when warmup finishes"""
    self.warmup_end = True if args.warmup_updates <= 0 else False
    """ initial learning rate"""
    """this self.lr is used only during init and/or warm up period"""
    self.lr = args.warmup_init_lr
    self.optimizer.set_lr(self.lr)

@staticmethod
def add_args(parser):
    """Add arguments to the parser for this LR scheduler."""
    # fmt: off
    parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                        help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
    parser.add_argument('--lr-patience', default=0, type=float, metavar='N',
                        help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--lr-threshold', default=1e-4, type=float, metavar='LT',
                        help='Threshold for measuring the new optimum, \
                        to only focus on significant changes')
    parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                        help='warmup the learning rate linearly for the first N updates')
    parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                        help='initial learning rate during warmup phase; default is args.lr')
    # fmt: on

ReduceLROnPlateau.__init__ = __init__
ReduceLROnPlateau.add_args = add_args

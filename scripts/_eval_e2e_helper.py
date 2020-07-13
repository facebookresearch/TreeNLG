# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import os

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from shutil import copyfile


copyfile('scripts/tmp/ref', 'scripts/tmp/ref_orig')
copyfile('scripts/tmp/hyp', 'scripts/tmp/hyp_orig')

class OpenFiles():
    def __init__(self):
        self.files = []
    def open(self, file_name, mode):
        f = open(file_name, mode)
        self.files.append(f)
        return f
    def close(self):
        list(map(lambda f: f.close(), self.files))

files = OpenFiles()
fsrc = files.open('scripts/tmp/src', 'r')
fref_orig = files.open('scripts/tmp/ref_orig', 'r')
fhyp_orig = files.open('scripts/tmp/hyp_orig', 'r')
fref = files.open('scripts/tmp/ref', 'w')
fhyp = files.open('scripts/tmp/hyp', 'w')

src_dict = defaultdict(lambda: [set(),list()])
for i, (src, hyp, ref) in enumerate(zip(fsrc, fhyp_orig, fref_orig)):
    src_dict[src][0].add(hyp)
    src_dict[src][1].append(ref)

for hyps, refs in src_dict.values():
    hyps = [h.split() for h in hyps]
    refs = [r.split() for r in refs]
    best_hyp = None
    best_bleu = 0.0
    for hyp in hyps:
        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
            bleu = sentence_bleu(refs, hyp)
        if bleu > best_bleu:
            best_hyp = hyp
            best_bleu = bleu
    fhyp.write(' '.join(best_hyp) + '\n')
    for ref in refs:
        fref.write(' '.join(ref) + '\n')
    fref.write('\n')

files.close()

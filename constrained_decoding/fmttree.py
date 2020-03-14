#!/usr/bin/env python

import argparse


def fmttree(tree: str, hide_terminals=False):
    indent = 0
    noninitial = False
    for tok in tree.split(' '):
        if tok.startswith('['):
            print('\n' * noninitial + '    ' * indent + tok, end='')
            indent += 1
            newline = False
            noninitial = True
        elif tok == ']':
            indent -= 1
            assert indent >= 0
            if newline:
                print('\n' + '    ' * indent + ']', end='')
            else:
                print(' ' + ']', end='')
            newline = True
        else:
            assert noninitial
            if not hide_terminals:
                print(' ' + tok, end='')
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="format linearized tree")
    parser.add_argument('tree')
    args = parser.parse_args()
    fmttree(args.tree)

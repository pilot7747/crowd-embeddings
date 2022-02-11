import sys
import pytorch_lightning as pl
from argparse import ArgumentParser


def parse_args(args):
    from benchmarks.benchmarks.benchmark import Benchmark
    parser = ArgumentParser(args)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Benchmark.add_argparse_args(parser, args=args)
    args = parser.parse_args(args)
    return args

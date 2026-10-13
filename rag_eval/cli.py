"""Command-line entry point: `python -m rag_eval <command>`."""

from __future__ import annotations

import argparse

from rag_eval.commands import check_eval_set, compare, failures, index, parse, run, search, sweep

COMMANDS = (parse, check_eval_set, index, search, run, sweep, compare, failures)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_eval")
    sub = p.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

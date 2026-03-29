#!/usr/bin/env python3
"""Fail when MiniProgram source files contain UTF-8 BOM."""

from __future__ import annotations

import argparse
import pathlib
import sys


DEFAULT_EXTENSIONS = {".js", ".json", ".wxml", ".wxss"}
UTF8_BOM = b"\xef\xbb\xbf"


def iter_files(root: pathlib.Path, extensions: set[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return files


def has_bom(path: pathlib.Path) -> bool:
    with path.open("rb") as f:
        head = f.read(3)
    return head == UTF8_BOM


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MiniProgram files are UTF-8 without BOM.")
    parser.add_argument(
        "--root",
        default=str(pathlib.Path(__file__).resolve().parents[1]),
        help="Root directory to scan (default: wechat_miniprogram).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions to check.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    failures = [path for path in iter_files(root, extensions) if has_bom(path)]
    if failures:
        print("Found UTF-8 BOM in MiniProgram files:", file=sys.stderr)
        for file_path in failures:
            print(f"- {file_path}", file=sys.stderr)
        return 1

    print(f"Encoding check passed: no BOM under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


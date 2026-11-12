"""Command line: ``affinage <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="affinage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser(
        "convert-hypersigma", help="convert the pinned HyperSIGMA snapshots to safetensors"
    )
    convert.add_argument("--spatial", type=Path, required=True)
    convert.add_argument("--spectral", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)

    convert_htc = commands.add_parser(
        "convert-htc", help="convert the pinned HTC organ-segmentation model to safetensors"
    )
    convert_htc.add_argument("--archive", type=Path, required=True)
    convert_htc.add_argument("--out", type=Path, required=True)

    train = commands.add_parser("train", help="run one configured customisation")
    train.add_argument("config", type=Path)

    export = commands.add_parser("export", help="export a run as merged weights with a model card")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--converted-dir", type=Path, default=None)

    verify = commands.add_parser("verify", help="check a record's outputs against their checksums")
    verify.add_argument("record", type=Path)

    synthetic = commands.add_parser(
        "synthetic", help="write a small synthetic dataset for a dry run"
    )
    synthetic.add_argument("out", type=Path)

    args = parser.parse_args(argv)
    command = ["affinage", *argv]

    if args.command == "verify":
        from fonderie_affinage.record import verify_outputs

        problems = verify_outputs(args.record)
        for problem in problems:
            print(problem)
        print("record verified" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if args.command == "synthetic":
        from fonderie_affinage.hsi.synthetic import write_synthetic_dataset

        print(write_synthetic_dataset(args.out))
        return 0
    if args.command == "convert-hypersigma":
        from fonderie_affinage.models.hypersigma import convert

        print(convert(args.spatial, args.spectral, args.out, command))
        return 0
    if args.command == "convert-htc":
        from fonderie_affinage.models.htc import convert as convert_htc_model

        print(convert_htc_model(args.archive, args.out, command))
        return 0
    if args.command == "train":
        from fonderie_affinage.config import load_config
        from fonderie_affinage.train import run_training

        print(run_training(load_config(args.config), command))
        return 0
    from fonderie_affinage.export import export_run

    print(export_run(args.run_dir, args.out, command, args.converted_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

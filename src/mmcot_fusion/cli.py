"""Command-line interface: ``mmcot-fusion {download,run}``."""

from __future__ import annotations

import argparse
import dataclasses


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmcot-fusion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download ScienceQA data, features, checkpoints")
    p_dl.add_argument("--data-root", default="data")
    p_dl.add_argument("--features-root", default="vision_features")
    p_dl.add_argument("--models-root", default="models")
    p_dl.add_argument("--img-types", nargs="+", default=["detr"])
    p_dl.add_argument("--checkpoints", choices=["none", "base", "large"], default="base")

    p_run = sub.add_parser("run", help="train or evaluate one stage")
    from .training.run import RunConfig

    for field in dataclasses.fields(RunConfig):
        name = "--" + field.name.replace("_", "-")
        type_str = str(field.type)
        if "bool" in type_str:
            p_run.add_argument(name, action="store_true")
        elif "int" in type_str:
            p_run.add_argument(name, default=field.default, type=int)
        elif "float" in type_str:
            p_run.add_argument(name, default=field.default, type=float)
        else:
            p_run.add_argument(name, default=field.default, type=str)

    args = parser.parse_args()

    if args.command == "download":
        from .data.download import (
            download_checkpoints,
            download_scienceqa,
            download_vision_features,
            verify_layout,
        )

        download_scienceqa(args.data_root)
        download_vision_features(args.features_root, tuple(args.img_types))
        if args.checkpoints != "none":
            download_checkpoints(args.models_root, args.checkpoints)
        verify_layout(args.data_root, args.features_root)
    elif args.command == "run":
        from .training.run import RunConfig, run

        field_names = {f.name for f in dataclasses.fields(RunConfig)}
        cfg_kwargs = {
            k: v for k, v in vars(args).items() if k in field_names and v is not None
        }
        # optional str fields arrive as the string "None" defaults; normalise
        for key, value in list(cfg_kwargs.items()):
            if value == "None" or value is None:
                cfg_kwargs[key] = None
        cfg = RunConfig(**cfg_kwargs)
        run(cfg)


if __name__ == "__main__":
    main()

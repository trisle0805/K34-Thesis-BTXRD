"""Train one BTXRD experiment configuration."""

from __future__ import annotations

import argparse
import json

from btxrd.config import load_config
from btxrd.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    result = train(load_config(args.config, args.set))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

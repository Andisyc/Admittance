"""Run explicitly on the remote training server; downloads data without training."""

import argparse
import json

from unilab.data.amass_download import prepare_amass_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Data directory on the training server")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare_amass_subset(args.output_dir, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()

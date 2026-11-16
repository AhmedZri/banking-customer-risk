import argparse
import json
from pathlib import Path
import pandas as pd
from .data import download, synthetic_customers
from .training import train


def main():
    parser = argparse.ArgumentParser(description="Academic banking risk platform")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--output", default="data/uci.csv")
    for name in ["train", "demo"]:
        p = sub.add_parser(name)
        p.add_argument("--csv", default="data/uci.csv")
        p.add_argument("--output", required=True)
        p.add_argument("--quick", action="store_true")
        p.add_argument("--evaluate-test", action="store_true")
        p.add_argument("--skip-explain", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(args.output), indent=2))
        return
    source = "synthetic" if args.command == "demo" else "UCI 350"
    if args.command == "demo":
        df = synthetic_customers()
    else:
        manifest = Path(args.csv).with_suffix(".manifest.json")
        if not manifest.exists():
            parser.error("Run download first to create the provenance manifest")
        import hashlib

        info = json.loads(manifest.read_text())
        if info["sha256"] != hashlib.sha256(Path(args.csv).read_bytes()).hexdigest():
            parser.error("Dataset checksum mismatch")
        df = pd.read_csv(args.csv)
    result = train(
        df,
        args.output,
        source=source,
        quick=args.quick,
        explain=not args.skip_explain,
        evaluate_test=args.evaluate_test,
    )
    print(
        json.dumps(
            {
                "selected_model": result["selected_model"],
                "test": result["test"]["selected"] if result["test"] else "sealed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

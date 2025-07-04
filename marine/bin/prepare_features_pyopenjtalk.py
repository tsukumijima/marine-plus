import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm

from marine.data.dictionary_downloader import download_and_apply_dictionaries
from marine.logger import getLogger
from marine.utils.openjtalk_util import convert_open_jtalk_node_to_feature
from marine.utils.util import load_json_corpus


# download and apply OpenJTalk dictionaries
download_and_apply_dictionaries()

logger = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Special format txt format data to json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_path", type=Path, help="Input path or directory")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def extract_feature(script_id, text):
    features = {"script_id": script_id, "nodes": []}

    try:
        from pyopenjtalk import run_frontend
    except BaseException:
        raise ImportError(
            'Please install pyopenjtalk by `pip install -e ".[dev,pyopenjtalk]"`'
        )

    # drop full-context label
    nodes = run_frontend(text)
    features["nodes"] = convert_open_jtalk_node_to_feature(nodes)

    return features


def _sort_corpus_by_script_id(corpus):
    return list(sorted(corpus, key=lambda x: x["script_id"]))


def entry(argv=sys.argv):
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(args)

    # Process
    n_jobs = min(cpu_count(), args.n_jobs)

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    # Load corpus
    corpus = load_json_corpus(args.in_path)

    if n_jobs > 1:
        logger.info(f"Processing {len(corpus):,} scripts with {n_jobs} jobs")
        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(
                    extract_feature,
                    script["script_id"],
                    script["surface"],
                )
                for script in corpus
            ]
            corpus = [
                future.result()
                for future in tqdm(
                    futures, desc="Convert corpus to feature", leave=False
                )
            ]
    else:
        logger.info(f"Processing {len(corpus):,} scripts in a single thread")
        corpus = [
            extract_feature(script["script_id"], script["surface"]) for script in corpus
        ]

    # corpus = _sort_corpus_by_script_id(corpus)

    output_path = args.out_dir / "feature.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(corpus, file, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    sys.exit(entry())

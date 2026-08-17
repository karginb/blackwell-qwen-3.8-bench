#!/usr/bin/env python3
"""
Fetch an evaluation corpus and write it out as the plain text llama-perplexity wants.

WHY WIKITEXT-2
Perplexity is meaningless as an absolute number -- "PPL = 7.3" says nothing on its
own. It only carries information as a comparison: same model, same text, different
quantization. wikitext-2-raw is the corpus almost every published quantization
comparison uses, so measuring on it keeps our numbers comparable to other people's
instead of stranded in a private scale.

Caveat worth stating in any writeup: wikitext is English Wikipedia prose. A quant
that holds up here can still degrade badly on code, on structured output, or in
another language. Perplexity is a cheap first signal, not a verdict -- llama.cpp
also ships --hellaswag / --winogrande / --multiple-choice for task-level checks.

USAGE
    python3 bench/prepare_corpus.py                     # wikitext-2 test split
    python3 bench/prepare_corpus.py --split validation
"""

import argparse
import os
import sys

REPO = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--out", default=None, help="output text file")
    args = p.parse_args()

    out = args.out or f"bench/data/wikitext-2-raw-{args.split}.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        sys.exit(f"missing dependency: {e}\n"
                 f"install with: pip install pyarrow huggingface_hub")

    remote = f"{CONFIG}/{args.split}-00000-of-00001.parquet"
    print(f"downloading {REPO}/{remote} ...")
    local = hf_hub_download(repo_id=REPO, filename=remote, repo_type="dataset")

    table = pq.read_table(local)
    rows = table.column("text").to_pylist()

    # The dataset stores one line per row, newlines already included. Joining with
    # an empty string reproduces the original continuous document, which is what
    # the standard perplexity protocol tokenizes.
    text = "".join(rows)

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    chars = len(text)
    words = len(text.split())
    print(f"wrote {out}")
    print(f"  {chars:,} characters, ~{words:,} words, {len(rows):,} rows")
    print(f"  roughly {words * 4 // 3:,} tokens -> about {words * 4 // 3 // 512:,} "
          f"chunks of 512 at llama-perplexity's default context")


if __name__ == "__main__":
    main()

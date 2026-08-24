"""Build a word-level KenLM training corpus from ASR manifests.

The corpus text is normalized to exactly the acoustic model's output space, so the
LM scores words the decoder can actually emit. Emits corpus.txt + lm.vocab and prints
the lmplz/build_binary commands; pass --kenlm_bin to run them.

    python build_kenlm.py --manifests /path/train.jsonl ... --out_dir ./lm_uz \
        --model_name multilingual_ctc --order 5 --prune 0 0 1

Note: a KenLM trained over BPE units (NeMo `encoding: bpe`) is NOT usable here —
GigaAM decodes charwise, so the LM must be word-level over normalized text.
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Set

from nemo_to_tsv import normalize_orthography

import gigaam
from gigaam.utils import normalize_raw_text


def iter_texts(path: Path) -> Iterator[str]:
    """Yield raw transcriptions from a NeMo .jsonl or a GigaAM .tsv manifest."""
    if path.suffix == ".jsonl" or path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line).get("text") or ""
    elif path.suffix == ".tsv":
        import csv

        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                yield row.get("transcription") or ""
    else:
        raise ValueError(f"Unsupported manifest type: {path} (want .jsonl or .tsv)")


def normalize(text: str, vocab: Set[str]) -> str:
    """Normalize to the model's output space.

    Order matters: the macron letters are alphanumeric, so normalize_raw_text passes
    them through untouched and the vocab filter below would then delete them outright
    (tōrt -> trt). Mapping them to o'/g' first keeps the word intact.
    """
    text = normalize_orthography(text)
    text = normalize_raw_text(text)
    return "".join(c for c in text if c in vocab)


def resolve_vocab(args) -> Set[str]:
    src = args.checkpoint or args.model_name
    model = gigaam.load_model(src, device="cpu")
    tokenizer = model.decoding.tokenizer
    if not getattr(tokenizer, "charwise", False):
        raise SystemExit(f"'{src}' is not a charwise model; a word-level LM needs one.")
    return set(tokenizer.vocab)


def check_leakage(manifests: List[Path], buckets_dir: str, allow: bool) -> None:
    bucket_stems = {p.stem for p in Path(buckets_dir).glob("*.tsv")}
    clashes = [m for m in manifests if m.stem in bucket_stems]
    if clashes and not allow:
        raise SystemExit(
            "Refusing to train an LM on eval-set transcripts (would fake a big WER "
            f"win): {', '.join(m.name for m in clashes)}\nPass --allow_eval_leak to "
            "override."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", required=True, help=".jsonl or .tsv")
    p.add_argument("--out_dir", default="./lm_corpus")
    p.add_argument("--model_name", default="multilingual_ctc", help="vocab source")
    p.add_argument(
        "--checkpoint", default=None, help="vocab source (overrides --model_name)"
    )
    p.add_argument("--order", type=int, default=5, help="n-gram order for lmplz")
    p.add_argument("--prune", nargs="+", type=int, default=[0, 0, 1])
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="drop duplicate lines; off by default because it distorts the n-gram "
        "frequencies the LM estimates",
    )
    p.add_argument(
        "--buckets_dir", default="../buckets", help="eval sets, for the leak guard"
    )
    p.add_argument("--allow_eval_leak", action="store_true")
    p.add_argument(
        "--kenlm_bin", default=None, help="dir with lmplz/build_binary; runs them"
    )
    p.add_argument(
        "--binary", action="store_true", help="also build the .bin (needs --kenlm_bin)"
    )
    p.add_argument("--memory", default="40%", help="lmplz -S")
    p.add_argument("--tmpdir", default="/tmp", help="lmplz -T")
    args = p.parse_args()

    manifests = [Path(m) for m in args.manifests]
    missing = [m for m in manifests if not m.is_file()]
    if missing:
        raise SystemExit(f"Missing: {', '.join(str(m) for m in missing)}")
    check_leakage(manifests, args.buckets_dir, args.allow_eval_leak)

    vocab = resolve_vocab(args)
    print(f"Vocab: {len(vocab)} chars from {args.checkpoint or args.model_name}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"

    unigrams: Counter = Counter()
    n_in = n_out = n_empty = n_dup = 0
    seen: Set[str] = set()
    with open(corpus_path, "w", encoding="utf-8") as out:
        for manifest in manifests:
            rows = 0
            for raw in iter_texts(manifest):
                n_in += 1
                line = normalize(raw, vocab)
                if not line:
                    n_empty += 1
                    continue
                if args.dedupe:
                    if line in seen:
                        n_dup += 1
                        continue
                    seen.add(line)
                out.write(line + "\n")
                unigrams.update(line.split())
                rows += 1
                n_out += 1
            print(f"  {rows:>8} lines  {manifest}")

    vocab_path = out_dir / "lm.vocab"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, _ in unigrams.most_common():
            f.write(word + "\n")

    stats = {
        "lines_in": n_in,
        "lines_out": n_out,
        "lines_empty": n_empty,
        "lines_dup_dropped": n_dup,
        "tokens": sum(unigrams.values()),
        "unique_words": len(unigrams),
        "top_words": unigrams.most_common(20),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"\n=> {corpus_path}: {n_out} lines, {stats['tokens']} tokens, "
        f"{stats['unique_words']} unique words ({n_empty} empty dropped"
        + (f", {n_dup} dups dropped" if args.dedupe else "")
        + f")\n=> {vocab_path}: unigram list (pass as --unigrams_path)"
    )

    arpa = out_dir / "lm.arpa"
    binary = out_dir / "lm.bin"
    prune = " ".join(str(x) for x in args.prune)
    lmplz_cmd = (
        f"lmplz -o {args.order} --prune {prune} -S {args.memory} -T {args.tmpdir} "
        f"< {corpus_path} > {arpa}"
    )
    build_cmd = f"build_binary trie {arpa} {binary}"

    if not args.kenlm_bin:
        print(
            "\nNext (lmplz is not bundled with the pip kenlm wheel — build it from "
            "kenlm sources or use a machine that has it):\n"
            f"  {lmplz_cmd}\n  {build_cmd}\n"
            "Then decode with:\n"
            f"  python eval_buckets.py --checkpoint ... --lm_path {binary} "
            f"--unigrams_path {vocab_path} --alpha 0.5 --beta 1.5"
        )
        return

    lmplz = Path(args.kenlm_bin) / "lmplz"
    if not lmplz.is_file():
        raise SystemExit(f"Not found: {lmplz}")
    print(f"\nRunning: {lmplz_cmd}")
    with open(corpus_path) as fin, open(arpa, "w") as fout:
        subprocess.run(
            [
                str(lmplz),
                "-o",
                str(args.order),
                "--prune",
                *map(str, args.prune),
                "-S",
                args.memory,
                "-T",
                args.tmpdir,
            ],
            stdin=fin,
            stdout=fout,
            check=True,
        )
    print(f"=> {arpa}")
    if args.binary:
        print(f"Running: {build_cmd}")
        subprocess.run(
            [
                str(Path(args.kenlm_bin) / "build_binary"),
                "trie",
                str(arpa),
                str(binary),
            ],
            check=True,
        )
        print(f"=> {binary}")


if __name__ == "__main__":
    main()

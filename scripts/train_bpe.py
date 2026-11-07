"""训练 WMT14 英德任务用的共享 BPE 词表。

对照《Attention Is All You Need》§5.1：
    "Sentences were encoded using byte-pair encoding, which has a
     shared source-target vocabulary of about 37000 tokens."

关键点
------
1. **共享词表**：英文和德语喂给同一个 trainer，共用一张 37,000 的表。
   好处是 encoder / decoder 可以共享 embedding 权重，id 空间也统一。
2. **不是按空格分词**：BPE 从字符出发，反复合并最高频的相邻符号对，
   直到词表达到 37000。罕见词会被切成子词，几乎不会 OOV。
3. **特殊 token 固定占前 4 个 id**，训练/推理/保存权重时顺序必须一致：
       0=<pad>  1=<unk>  2=<bos>  3=<eos>

用法
----
    # 先用小样本快速验证流程（几十秒）
    python scripts/train_bpe.py --max-lines 200000

    # 正式跑全量 450 万句对
    python scripts/train_bpe.py
"""

from __future__ import annotations

import argparse
import os
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# 顺序即 id：不要调换，改了就和已存的模型权重对不上
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

DEFAULT_VOCAB = 37000


def iter_corpus(paths: list[str], max_lines: int | None = None):
    """逐行产出句子。中英德共用，所以两个文件直接拼在一个迭代器里。"""
    for path in paths:
        n = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line
                    n += 1
                    if max_lines and n >= max_lines:
                        break


def count_lines(paths: list[str], max_lines: int | None) -> int:
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for i, _ in enumerate(fh, 1):
                if max_lines and i > max_lines:
                    total += max_lines
                    break
            else:
                total += i
    return total


def build_tokenizer(vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # Whitespace 预切分：只在空格处切开，单词内部的 BPE 合并交给 trainer 学
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser(description="训练共享 BPE 词表")
    ap.add_argument("--data-dir", default="data/wmt14_en_de")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="每个文件最多取多少行（调试用，默认全量）",
    )
    ap.add_argument("--out", default=None, help="输出 json 路径")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    data_dir = args.data_dir
    train_paths = [os.path.join(data_dir, "train.en"), os.path.join(data_dir, "train.de")]
    for p in train_paths:
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}，请先运行 scripts/download_wmt14.py")

    out = args.out or os.path.join(data_dir, "bpe37k_shared.json")

    print(f"语料：{train_paths[0]}  +  {train_paths[1]}")
    total = count_lines(train_paths, args.max_lines)
    print(f"训练句数：{total:,}（每边最多 {args.max_lines or '全部'} 行）")
    print(f"目标词表：{args.vocab_size:,}  tokens，共享 en/de\n")

    tok = build_tokenizer(args.vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        end_of_word_suffix="</w>",
    )

    t0 = time.time()
    tok.train_from_iterator(
        iter_corpus(train_paths, args.max_lines), trainer=trainer, length=total
    )
    print(f"\n训练耗时 {time.time() - t0:.1f}s，实际词表大小 {tok.get_vocab_size():,}")

    # 注意：这里**不要** enable_padding。
    # 一旦写进 json，encode_batch 会自动把一批句子补到该批最长，
    # 导致存下来的 id 序列里混进大量假 <pad>，长度还全都一样。
    # padding 是批处理的职责，交给 utils/data.py 的 collate_fn 做。
    tok.save(out)
    print(f"已保存：{out}")

    # ---- 演示：看真实句子是怎么变成 id 的 -------------------------------
    print("\n" + "=" * 68)
    print("编码示例（词 → 子词 → id）")
    print("=" * 68)
    demo = [
        "Resumption of the session",
        "Wiederaufnahme der Sitzungsperiode",
        "The Enhancer and HDR modes are useful in cases when a photo has an uneven exposure.",
    ]
    for s in demo:
        enc = tok.encode(s)
        print(f"\n原文: {s}")
        print(f"子词: {' '.join(enc.tokens)}")
        print(f"id  : {enc.ids}")
        # 加 bos/eos 后才是真正喂给 decoder 的序列
        with_special = [BOS_ID] + enc.ids + [EOS_ID]
        print(f"含特殊符: {with_special}")

    vocab = tok.get_vocab()
    print("\n特殊 token 的 id：")
    for t in SPECIAL_TOKENS:
        print(f"  {t:<6} -> {vocab[t]}")


if __name__ == "__main__":
    main()

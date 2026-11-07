"""BLEU 评测：加载 checkpoint，在 newstest2014 上做贪心解码，用 sacrebleu 打分。

用法（项目根目录）：

    # 完整测试集
    E:/Aconda/envs/ultralytics/python.exe -m scripts.evaluate --ckpt checkpoint/xxx/last_model.pth

    # 先跑 50 句确认能跑通
    E:/Aconda/envs/ultralytics/python.exe -m scripts.evaluate --limit 50

    # 换个 checkpoint / 换验证集
    E:/Aconda/envs/ultralytics/python.exe -m scripts.evaluate --split valid --ckpt ...

说明：
    * 贪心解码（beam=1）。论文的 27.3 是 beam=4 + length penalty 0.6 的结果，
      贪心一般低 1~3 分，属于正常差距。
    * 目标侧用 <bos> 开头，遇到 <eos> 停；超过 --max-len 强制截断。
    * 保存假设到 --out（默认 predictions.txt），方便事后用其他工具再算。
"""

import argparse
import os
import re
import time

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from Transformer.model import Transformer

BOS, EOS, PAD, UNK = 2, 3, 0, 1
VOCAB = 37000

SPLITS = {
    "test": ("data/wmt14_en_de/test.en", "data/wmt14_en_de/test.de"),
    "valid": ("data/wmt14_en_de/valid.en", "data/wmt14_en_de/valid.de"),
}

# BPE 的 </w> 解码会在每个 token 后补空格，于是标点、撇号、连字符前面都会多一个空格：
#   "Gutach: Noch"      -> "Gutach : Noch"
#   "Gutacht's"         -> "Gutacht ' s"
#   "33-Fußgängerampel" -> "33 - Fußgängerampel"
# 参考译文里没有这些空格。下面的规则把常见几类还原回去。
_RE_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%\)\]\}»”])")
_RE_SPACE_AFTER = re.compile(r"([\(\[\{«„“])\s+")
_RE_APOS = re.compile(r"(?<=\w)\s+'\s*(?=s\b|t\b|d\b|m\b|re\b|ve\b|ll\b)")
# 只合并 Gutacht ' s / don ' t 这类缩写，避免把 the 'Grub' residential 粘成 the'Grub'residential
_RE_SLASH = re.compile(r"\s*/\s*")                 # Sulzbachweg / Kirchstrasse -> .../...
_RE_HYPHEN_NUM = re.compile(r"(\d)\s+-\s+")        # 33 - Fußgängerampel -> 33-...


def detok(text):
    text = _RE_SPACE_BEFORE.sub(r"\1", text)
    text = _RE_SPACE_AFTER.sub(r"\1", text)
    text = _RE_APOS.sub("'", text)
    text = _RE_SLASH.sub("/", text)
    text = _RE_HYPHEN_NUM.sub(r"\1-", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def load_model(ckpt_path, device, n_layers=6, heads=8, dims=512):
    model = Transformer(N=n_layers, heads=heads, dims=dims,
                        des_vocab_size=VOCAB, tie_weights=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()

    meta = []
    if "epoch" in ckpt:
        meta.append(f"tokens={int(ckpt['epoch']):,}")
    if ckpt.get("loss"):
        meta.append(f"valid_loss={ckpt['loss']:.4f}")
    print(f"已加载 {ckpt_path}" + ("  (" + ", ".join(meta) + ")" if meta else ""))

    # weight tying 是否真的生效
    tied = (model.Linear.weight.data_ptr() == model.Embedding.Embedding.weight.data_ptr())
    print(f"参数量 {sum(p.numel() for p in model.parameters()):,}  weight tying={tied}")
    return model


def encode_src(tok, lines, device, max_seq_len):
    """把一批句子编码成 (B,S) 和 (B,S) 的 key_padding_mask（0 保留 / -inf 屏蔽）。"""
    ids_list = []
    for line in lines:
        ids = tok.encode(line).ids[:max_seq_len]
        ids_list.append(ids)
    S = max(1, max(len(x) for x in ids_list))

    src_ids = torch.full((len(lines), S), PAD, dtype=torch.long)
    for r, ids in enumerate(ids_list):
        if ids:
            src_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    src_ids = src_ids.to(device)

    mask = torch.zeros(len(lines), S, dtype=torch.float32, device=device)
    mask[src_ids == PAD] = float("-inf")
    return src_ids, mask


def get_logits(model, src_ids, ys, src_mask, fp16):
    """整句重算一次前向（没有 KV cache）。"""
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=fp16 and src_ids.device.type == "cuda"):
        out = model(src_ids, ys, src_mask, None)
    return out[:, -1].float()          # 只要最后一个位置 (B, V)


@torch.no_grad()
def beam_decode(model, tok, src_lines, device, batch_size=16, max_len=256,
                src_max_len=400, fp16=True, fix_punct=True, beam=4, lenpen=0.6,
                max_len_extra=50):
    """束搜索解码。beam=1 时等价于贪心。

    思路：把每句展开成 beam 份，逐步 topk。已经生成 <eos> 的 beam 冻结分数
    （logp 置 0）、长度不再增长，等最后统一按长度归一化分数排序。
    """
    hyps = []
    for i in tqdm(range(0, len(src_lines), batch_size), unit="batch", ncols=90):
        chunk = src_lines[i:i + batch_size]
        src_ids, src_mask = encode_src(tok, chunk, device, src_max_len)
        B = src_ids.size(0)
        # 论文 §6：max_len = 源句长度 + 50
        limit = min(max_len, int(src_ids.size(1)) + max_len_extra)
        V = VOCAB

        src_e = src_ids.repeat_interleave(beam, 0)          # (B*beam, S)
        mask_e = src_mask.repeat_interleave(beam, 0)

        ys = torch.full((B, beam, 1), BOS, dtype=torch.long, device=device)
        scores = torch.full((B, beam), -1e9, device=device)
        scores[:, 0] = 0.0                                   # 只有第一条 beam 起步
        lens = torch.zeros(B, beam, device=device)            # 已生成 token 数
        fin = torch.zeros(B, beam, dtype=torch.bool, device=device)

        for _ in range(limit):
            logits = get_logits(model, src_e, ys.reshape(B * beam, -1), mask_e, fp16)
            logp = torch.log_softmax(logits, dim=-1).view(B, beam, V)
            # 已结束的 beam 不再累积分数，后续追加的 token 会被截断时丢掉
            logp = logp.masked_fill(fin[:, :, None], 0.0)

            total = (scores[:, :, None] + logp).view(B, beam * V)
            top_s, top_i = total.topk(beam, dim=-1)          # (B, beam)
            prev = top_i // V
            nxt = top_i % V

            old_len = lens.gather(1, prev)
            old_fin = fin.gather(1, prev)
            ys = ys.gather(1, prev[:, :, None].expand(-1, -1, ys.size(-1)))
            ys = torch.cat([ys, nxt[:, :, None]], dim=-1)

            scores = top_s
            fin = old_fin | (nxt == EOS)
            lens = torch.where(old_fin, old_len, old_len + 1)

            if bool(fin.all()):
                break

        # 长度归一化：score / ((5+len)/6)^α
        norm = scores / (((5.0 + lens) / 6.0) ** lenpen)
        best = norm.argmax(dim=-1)                           # (B,)
        picked = ys.gather(1, best[:, None, None].expand(-1, 1, ys.size(-1)))[:, 0]

        for row in picked[:, 1:].cpu().tolist():             # 去掉 <bos>
            ids = []
            for t in row:
                if t == EOS:
                    break
                ids.append(t)
            hyps.append(tok.decode(ids, skip_special_tokens=True).strip())

    if fix_punct:
        hyps = [detok(h) for h in hyps]
    return hyps


@torch.no_grad()
def greedy_decode(model, tok, src_lines, device, batch_size=32, max_len=256,
                  src_max_len=400, fp16=True, fix_punct=True, max_len_extra=50):
    """逐批贪心解码，返回译文列表（已 detokenize）。"""
    hyps = []
    for i in tqdm(range(0, len(src_lines), batch_size), unit="batch", ncols=90):
        chunk = src_lines[i:i + batch_size]
        src_ids, src_mask = encode_src(tok, chunk, device, src_max_len)
        B = src_ids.size(0)

        # 论文 §6：max_len = 源句长度 + 50
        limit = min(max_len, int(src_ids.size(1)) + max_len_extra)

        ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(limit):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                logits = model(src_ids, ys, src_mask, None)
            nxt = logits[:, -1].float().argmax(-1)          # (B,)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if done.all():
                break

        for row in ys[:, 1:].cpu().tolist():                # 去掉开头的 <bos>
            ids = []
            for t in row:
                if t == EOS:
                    break
                ids.append(t)
            hyps.append(tok.decode(ids, skip_special_tokens=True).strip())
    if fix_punct:
        hyps = [detok(h) for h in hyps]
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pth", help="checkpoint 路径，默认自动找最新的 last_model.pth")
    ap.add_argument("--split", default="valid", choices=list(SPLITS))
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=256, help="目标侧最大解码长度")
    ap.add_argument("--max-len-extra", type=int, default=50,
                    help="解码上限 = 源句长 + 该值（论文用 50）")
    ap.add_argument("--src-max-len", type=int, default=400, help="源句截断长度")

    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 句（调试用）")
    ap.add_argument("--beam", type=int, default=4, help="束宽，1=贪心，论文用 4")
    ap.add_argument("--lenpen", type=float, default=0.6, help="长度惩罚 α，论文 0.6")
    ap.add_argument("--out", default="predictions.txt")
    ap.add_argument("--show", type=int, default=5, help="打印前 N 个译文样例")
    ap.add_argument("--no-fix-punct", action="store_true", help="不做标点前的空格归一化")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file("data/wmt14_en_de/bpe37k_shared.json")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        cand = sorted([os.path.join(d, f) for d, _, fs in os.walk("checkpoint")
                       for f in fs if f == "last_model.pth"],
                      key=os.path.getmtime)
        if not cand:
            raise SystemExit("没找到 checkpoint，请用 --ckpt 指定")
        ckpt_path = cand[-1]
        print(f"未指定 --ckpt，自动使用最新的: {ckpt_path}")

    model = load_model(ckpt_path, device)

    src_file, ref_file = SPLITS[args.split]
    with open(src_file, encoding="utf-8") as f:
        src_lines = [x.rstrip("\n") for x in f]
    with open(ref_file, encoding="utf-8") as f:
        refs = [x.rstrip("\n") for x in f]
    n = min(len(src_lines), len(refs))
    src_lines, refs = src_lines[:n], refs[:n]
    if args.limit:
        src_lines, refs = src_lines[:args.limit], refs[:args.limit]
    print(f"{args.split} 集 {len(src_lines)} 句，batch={args.batch_size}，"
          f"device={device}，beam={args.beam}，lenpen={args.lenpen}")

    t0 = time.time()
    if args.beam <= 1:
        hyps = greedy_decode(model, tok, src_lines, device,
                             batch_size=args.batch_size, max_len=args.max_len,
                             src_max_len=args.src_max_len, fp16=not args.no_fp16,
                             fix_punct=not args.no_fix_punct,
                             max_len_extra=args.max_len_extra)
    else:
        hyps = beam_decode(model, tok, src_lines, device,
                           batch_size=args.batch_size, max_len=args.max_len,
                           src_max_len=args.src_max_len, fp16=not args.no_fp16,
                           fix_punct=not args.no_fix_punct,
                           beam=args.beam, lenpen=args.lenpen,
                           max_len_extra=args.max_len_extra)
    dt = time.time() - t0
    print(f"解码完成，用时 {dt:.1f}s（{dt/max(len(hyps),1)*1000:.0f} ms/句）")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    print(f"译文已保存: {args.out}")

    for k in range(min(args.show, len(hyps))):
        print(f"\n--- 第 {k+1} 句 ---")
        print(f"源  : {src_lines[k][:100]}")
        print(f"参考: {refs[k][:100]}")
        print(f"输出: {hyps[k][:100]}")

    try:
        import sacrebleu
    except ImportError:
        print("\n未安装 sacrebleu，跳过打分：pip install sacrebleu")
        return

    print()
    from sacrebleu.metrics import BLEU

    for tok_name in ("13a", "intl", "none"):
        metric = BLEU(tokenize=tok_name)          # 默认 case=mixed, smooth=exp
        score = metric.corpus_score(hyps, [refs])
        print(f"BLEU ({tok_name:>4} 分词): {score.score:6.2f}   {score}")
        if tok_name == "13a":
            print(f"      signature: {metric.get_signature()}")

    print("\n报告 BLEU 时请附上 signature，别人才能复现。"
          "\n参考：论文 base 模型在 newstest2014 上 BLEU 27.3"
          "（beam=4, length_penalty=0.6, 最后 5 个 checkpoint 平均）")


if __name__ == "__main__":
    main()

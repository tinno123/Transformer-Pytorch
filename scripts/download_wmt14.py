"""WMT 2014 英德(EN-DE) 数据集下载与整理脚本。

背景
----
本机无法访问 huggingface.co（返回 000），因此不走 `datasets.load_dataset("wmt14", "de-en")`，
改为直接从 WMT 官方源 www.statmt.org 下载原始语料并整理成纯文本平行语料。

产出
----
    <outdir>/train.en   <outdir>/train.de      # 训练集
    <outdir>/valid.en   <outdir>/valid.de      # 验证集 = newstest2013
    <outdir>/test.en    <outdir>/test.de       # 测试集 = newstest2014

用法
----
    # 冒烟测试（约 100MB，News Commentary v9 ≈ 24 万句对）
    python scripts/download_wmt14.py --level smoke

    # 完整集（约 1.5GB 下载，450 万句对，复现论文用）
    python scripts/download_wmt14.py --level full

    # 只下 Common Crawl，或只下 Europarl
    python scripts/download_wmt14.py --level custom --parts nc commoncrawl

可选参数
--------
    --outdir   输出目录，默认 data/wmt14_en_de
    --keep     保留下载的 .tgz 压缩包（默认解压后删除）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import time
import urllib.request
from html.parser import HTMLParser

BASE = "https://www.statmt.org"

# ---------------------------------------------------------------------------
# 下载源清单（WMT14 官方，en-de 可用部分）
# ---------------------------------------------------------------------------
PARTS = {
    # name: (url, 说明)
    "nc": (
        f"{BASE}/wmt14/training-parallel-nc-v9.tgz",
        "News Commentary v9（77MB，约 24 万句对）",
    ),
    "europarl": (
        f"{BASE}/wmt13/training-parallel-europarl-v7.tgz",
        "Europarl v7（628MB，约 190 万句对）",
    ),
    "commoncrawl": (
        f"{BASE}/wmt13/training-parallel-commoncrawl.tgz",
        "Common Crawl（876MB，约 240 万句对）",
    ),
    "dev": (
        f"{BASE}/wmt14/dev.tgz",
        "开发集 newstest2008~2013（17MB，验证集取 newstest2013）",
    ),
    "test": (
        f"{BASE}/wmt14/test-full.tgz",
        "测试集 newstest2014 清洗完整版（3.2MB）",
    ),
}

SMOKE_PARTS = ["nc", "dev", "test"]
FULL_PARTS = ["europarl", "commoncrawl", "nc", "dev", "test"]

# 临时下载目录名
TMP_DIRNAME = "_raw"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """urllib 下载进度条。"""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_d = downloaded / 1024 / 1024
        mb_t = total_size / 1024 / 1024
        bar = "#" * int(pct // 2.5).rjust(40, " ")
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_d:7.1f}/{mb_t:.1f} MB")
    else:
        sys.stdout.write(f"\r  已下载 {downloaded / 1024 / 1024:.1f} MB")
    sys.stdout.flush()


def remote_size(url: str) -> int:
    """用 HEAD 请求拿远端文件大小；拿不到返回 -1。"""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.headers.get("Content-Length") or -1)
    except Exception:  # noqa: BLE001
        return -1


def download(url: str, dst: str, retries: int = 3) -> str:
    """带进度条与断点续传的下载。

    注意：必须比对远端 Content-Length 才能判定"已下载完成"。
    只看"文件存在"会把中断留下的残片当成完整包，后续解压必然报
    EOFError: Compressed file ended before the end-of-stream marker。
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    total = remote_size(url)

    if total > 0 and os.path.exists(dst):
        size = os.path.getsize(dst)
        if size == total:
            print(f"  已存在且完整，跳过下载：{os.path.basename(dst)}")
            return dst
        if size > total:
            print(f"  文件异常（{size:,} > {total:,}），删除重下")
            os.remove(dst)
        else:
            print(f"  续传：已有 {size:,} / {total:,} 字节")

    for attempt in range(1, retries + 1):
        try:
            pos = os.path.getsize(dst) if os.path.exists(dst) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if pos:
                req.add_header("Range", f"bytes={pos}-")

            with urllib.request.urlopen(req, timeout=60) as resp, open(
                dst, "ab" if pos else "wb"
            ) as fh:
                # 服务器不支持断点续传时会回 200 并从头吐数据，此时必须重头写
                if pos and getattr(resp, "status", 206) != 206:
                    print("  服务器不支持断点续传，从头重下")
                    fh.seek(0)
                    fh.truncate()
                    pos = 0
                base = pos
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    base += len(chunk)
                    _progress_hook(base // (1024 * 256), 1024 * 256, total)
            print()

            if total > 0 and os.path.getsize(dst) < total:
                raise IOError(
                    f"下载不完整 {os.path.getsize(dst):,} / {total:,} 字节"
                )
            return dst
        except Exception as exc:  # noqa: BLE001
            print(f"\n  第 {attempt}/{retries} 次下载失败：{exc}")
            if attempt == retries:
                raise
            time.sleep(3)
    return dst


def safe_extract(tar_path: str, dest: str) -> list[str]:
    """安全解压，防路径穿越；返回解压出的文件路径列表。"""
    extracted = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()):
                continue
            name = member.name.replace("\\", "/")
            # 去掉绝对路径与 .. 段
            parts = [p for p in name.split("/") if p not in ("", ".", "..")]
            if not parts:
                continue
            member.name = "/".join(parts)
            target = os.path.join(dest, *parts)
            if not os.path.abspath(target).startswith(os.path.abspath(dest)):
                continue
            tar.extract(member, dest)
            if member.isfile():
                extracted.append(target)
    return extracted


class SGMLParser(HTMLParser):
    """从 WMT 的 .sgm 文件抽取 <seg id="n">文本</seg>。"""

    def __init__(self) -> None:
        super().__init__()
        self.segments: list[str] = []
        self._in_seg = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "seg":
            self._in_seg = True

    def handle_endtag(self, tag):
        if tag.lower() == "seg":
            self._in_seg = False

    def handle_data(self, data):
        if self._in_seg:
            text = data.strip()
            if text:
                self.segments.append(text)


def read_sgm(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        html = fh.read()
    parser = SGMLParser()
    parser.feed(html)
    return parser.segments


# str.splitlines() 还会切断 \v \f \x1c-\x1e \x85 \u2028 \u2029，
# 句内残留的 \r 同理，这些都会破坏"一行一句"的对齐，统一替换成空格。
_SPLIT_CHARS = re.compile(r"[\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+")


def clean_line(s: str) -> str:
    return _SPLIT_CHARS.sub(" ", s).strip()


def read_plain(path: str) -> list[str]:
    """读取平行语料。

    关键：**不能过滤空行**。en/de 两侧空行出现的位置不同，
    一旦各自独立过滤，后面的句子就会整体错位。
    这里保持行号一一对应，空行留到 align() 里按"对"剔除。
    """
    # newline="\n" 关闭通用换行翻译，避免句内 \r 被当成换行
    with open(path, "r", encoding="utf-8", errors="replace", newline="\n") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [clean_line(line) for line in lines]


def align(en: list[str], de: list[str]) -> tuple[list[str], list[str]]:
    """按行号配对，剔除任一侧为空的句子对，保证严格对齐。"""
    n = min(len(en), len(de))
    if len(en) != len(de):
        print(f"  ! 行数不一致 en={len(en):,} de={len(de):,}，截断到 {n:,}")
    en, de = en[:n], de[:n]
    keep = [(a, b) for a, b in zip(en, de) if a and b]
    dropped = n - len(keep)
    if dropped:
        print(f"  · 剔除含空句的句子对 {dropped:,} 条")
    return [a for a, _ in keep], [b for _, b in keep]


def write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    print(f"  写入 {path}  ({len(lines):,} 行)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def find_lang_files(files: list[str], lang: str, corpus: str | None = None) -> list[str]:
    """在解压出的文件中找出指定语言的平行文件。"""
    hits = []
    for f in files:
        name = os.path.basename(f)
        if corpus and corpus not in name:
            continue
        # 匹配 .en / .de 结尾，或 .en.sgm / .de.sgm
        if re.search(rf"[.]{lang}(\.sgm)?$", name):
            hits.append(f)
    return sorted(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并整理 WMT14 英德数据集")
    ap.add_argument(
        "--level",
        choices=["smoke", "full", "custom"],
        default="smoke",
        help="smoke=仅 News Commentary(快)；full=完整 450 万句对；custom=配合 --parts",
    )
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), help="自定义下载哪些部分")
    ap.add_argument("--outdir", default="data/wmt14_en_de", help="输出目录")
    ap.add_argument("--keep", action="store_true", help="保留 .tgz 压缩包")
    args = ap.parse_args()

    if args.level == "smoke":
        parts = SMOKE_PARTS
    elif args.level == "full":
        parts = FULL_PARTS
    else:
        if not args.parts:
            ap.error("--level custom 必须配合 --parts")
        parts = args.parts

    outdir = os.path.abspath(args.outdir)
    raw_dir = os.path.join(outdir, TMP_DIRNAME)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"输出目录：{outdir}")
    print(f"下载部分：{', '.join(parts)}\n")

    # ---- 1. 下载 + 解压 -----------------------------------------------------
    extracted: dict[str, list[str]] = {}
    for name in parts:
        url, desc = PARTS[name]
        dest = os.path.join(raw_dir, name)
        tgz = os.path.join(raw_dir, url.rsplit("/", 1)[-1])
        print(f"[{name}] {desc}")

        # 已解压过则直接复用，避免重复下载（完整集很大，这一步很关键）
        cached = []
        if os.path.isdir(dest):
            for root, _, fs in os.walk(dest):
                cached.extend(os.path.join(root, f) for f in fs)
        if cached:
            print(f"  复用已解压文件 {len(cached)} 个，跳过下载")
            extracted[name] = cached
            continue

        files: list[str] = []

        # 1) 若压缩包已存在（脚本下载的或手动放的），先试着直接解压。
        #    代理层不支持 Range、也不返回 Content-Length，无法事先判断完整性，
        #    所以"解压能否成功"就是唯一可靠的完整性校验。
        if os.path.exists(tgz) and os.path.getsize(tgz) > 0:
            try:
                print("  发现已下载的压缩包，直接解压 ...", end=" ")
                files = safe_extract(tgz, dest)
                print(f"共 {len(files)} 个文件")
            except Exception as exc:  # noqa: BLE001
                print(f"失败（{exc}）—— 残片，删除重下")
                os.remove(tgz)

        # 2) 没有可用的压缩包才真正下载
        if not files:
            download(url, tgz)
            try:
                print("  解压中 ...", end=" ")
                files = safe_extract(tgz, dest)
                print(f"共 {len(files)} 个文件")
            except Exception as exc:  # noqa: BLE001
                print(f"失败：{exc}")
                os.remove(tgz)
                raise

        extracted[name] = files
        if not args.keep:
            os.remove(tgz)

    # ---- 2. 整理训练集 -----------------------------------------------------
    train_src: list[str] = []
    train_tgt: list[str] = []

    for corpus_key, label in [
        ("europarl", "europarl"),
        ("commoncrawl", "commoncrawl"),
        ("nc", "news-commentary"),
    ]:
        if corpus_key not in extracted:
            continue
        files = extracted[corpus_key]
        en_files = [f for f in files if re.search(r"[.]en$", f) and "de-en" in os.path.basename(f)]
        de_files = [f for f in files if re.search(r"[.]de$", f) and "de-en" in os.path.basename(f)]
        if not en_files or not de_files:
            print(f"  跳过 {label}：未找到 de-en 平行文件")
            continue
        en = read_plain(en_files[0])
        de = read_plain(de_files[0])
        en, de = align(en, de)
        print(f"  {label}: {len(en):,} 句对")
        train_src.extend(en)
        train_tgt.extend(de)

    if train_src:
        print(f"训练集合计 {len(train_src):,} 句对")
        write_lines(os.path.join(outdir, "train.en"), train_src)
        write_lines(os.path.join(outdir, "train.de"), train_tgt)

    # ---- 3. 验证集：newstest2013 -------------------------------------------
    if "dev" in extracted:
        dev_files = extracted["dev"]
        en = de = None
        for f in dev_files:
            base = os.path.basename(f)
            if "newstest2013" in base:
                if base.endswith(".en"):
                    en = f
                elif base.endswith(".de"):
                    de = f
        if en and de:
            v_en, v_de = align(read_plain(en), read_plain(de))
            write_lines(os.path.join(outdir, "valid.en"), v_en)
            write_lines(os.path.join(outdir, "valid.de"), v_de)
        else:
            print("  警告：dev.tgz 中未找到 newstest2013，跳过验证集")

    # ---- 4. 测试集：newstest2014 -------------------------------------------
    if "test" in extracted:
        test_files = extracted["test"]
        src_en = ref_de = None
        for f in test_files:
            base = os.path.basename(f)
            if "newstest2014" not in base or "deen" not in base:
                continue
            if base.endswith("src.en.sgm"):
                src_en = f
            elif base.endswith("ref.de.sgm"):
                ref_de = f
        if src_en and ref_de:
            t_en, t_de = align(read_sgm(src_en), read_sgm(ref_de))
            write_lines(os.path.join(outdir, "test.en"), t_en)
            write_lines(os.path.join(outdir, "test.de"), t_de)
        else:
            print("  警告：test.tgz 中未找到 newstest2014-deen sgm，跳过测试集")

    print("\n完成。目录内容：")
    for f in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, f)
        if os.path.isfile(p):
            print(f"  {f:16s} {os.path.getsize(p) / 1024 / 1024:8.2f} MB")


if __name__ == "__main__":
    main()

"""
Paderborn 数据集下载重试脚本（带断点续传 + 重试 + 7-Zip 解压）
解决 utils/download_dataset.py 遇到 Zenodo SSL/网络中断直接崩溃的问题。

用法（在已激活的 venv 里）：
    python download_paderborn_retry.py --repo C:\\Users\\19185\\bearing-fault-diagnosis --bearings K001 KA04 KI04
    python download_paderborn_retry.py --repo ... --minimal
    python download_paderborn_retry.py --repo ... --minimal --insecure   # SSL 仍失败时可试
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://zenodo.org/records/15845309/files"
MINIMAL_SET = [
    'K001', 'K002', 'K003', 'K004', 'K005',
    'KA04', 'KA15', 'KA16', 'KA22', 'KA30',
    'KI04', 'KI14', 'KI16', 'KI18', 'KI21',
]
SEVEN_ZIP_CANDIDATES = [
    "7z",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def download_with_retry(url: str, dest: Path, retries: int = 10,
                        insecure: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=(15, 120), verify=not insecure) as r:
                if r.status_code == 416:
                    print(f"  {dest.name}: 已完成，跳过下载")
                    return True
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code == 200:
                    existing = 0
                    mode = "wb"
                else:
                    mode = "ab"
                total = int(r.headers.get("content-length", 0)) + existing
                with open(dest, mode) as f, tqdm(
                    total=total, initial=existing, unit="B", unit_scale=True,
                    desc=dest.name, leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"  {dest.name}: 下载完成")
            return True
        except Exception as e:
            print(f"  [{attempt}/{retries}] {dest.name} 失败：{e}")
            time.sleep(min(60, 5 * attempt))
    return False


def extract(rar_path: Path, mat_dir: Path) -> bool:
    mat_dir.mkdir(parents=True, exist_ok=True)
    for cmd in SEVEN_ZIP_CANDIDATES:
        try:
            r = subprocess.run(
                [cmd, "x", "-y", f"-o{mat_dir}", str(rar_path)],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                print(f"  解压成功：{cmd}")
                return True
            print(f"  {cmd} 解压失败：{r.stderr.strip() or r.stdout.strip()}")
        except FileNotFoundError:
            continue
    print("  找不到 7-Zip，请安装到 C:\\Program Files\\7-Zip")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis")
    p.add_argument("--bearings", nargs="*", default=None)
    p.add_argument("--minimal", action="store_true")
    p.add_argument("--insecure", action="store_true",
                   help="关闭 SSL 证书校验（仅用于 SSL 反复失败时）")
    p.add_argument("--retries", type=int, default=10)
    args = p.parse_args()

    bearings = args.bearings or (MINIMAL_SET if args.minimal else ["K001", "KA04", "KI04"])
    repo = Path(args.repo)
    rar_dir = repo / "paderborn_data" / "rar"
    mat_dir = repo / "paderborn_data" / "mat"

    if not repo.exists():
        raise SystemExit(f"仓库不存在：{repo}")

    print(f"轴承列表：{bearings}")
    for b in bearings:
        rar = rar_dir / f"{b}.rar"
        # 已有 .mat 就跳过
        if (mat_dir / b).exists() and any((mat_dir / b).rglob("*.mat")):
            print(f"{b}: 已存在，跳过")
            continue
        # 已有完整 rar 就直接解压，否则下载
        url = f"{BASE_URL}/{b}.rar?download=1"
        if rar.exists() and rar.stat().st_size > 0:
            print(f"{b}: 发现已有 {rar.name}，尝试直接解压")
        else:
            print(f"{b}: 下载 {url}")
            if not download_with_retry(url, rar, args.retries, args.insecure):
                print(f"{b}: 下载失败，稍后重跑本脚本会自动续传")
                continue
        if extract(rar, mat_dir):
            # 解压成功后删除 rar（想保留就注释掉这行）
            try:
                rar.unlink()
            except OSError:
                pass

    print("\n检查结果：")
    for b in bearings:
        files = list((mat_dir / b).rglob("*.mat")) if (mat_dir / b).exists() else []
        print(f"  {b}: {len(files)} 个 .mat")


if __name__ == "__main__":
    main()

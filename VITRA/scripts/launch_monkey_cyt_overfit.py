#!/usr/bin/env python3
"""Find monkey-cyt in 分布式训练空间 and launch VITRA overfit via qzcli exec."""
from __future__ import annotations

import argparse
import os
import subprocess

WS_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
WS_NAME = "分布式训练空间"
HOST_CANDIDATES = ("monkey-cyt", "monkey_cyt", "Monkey-cyt")
REPO = "/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/yangtao/Moneky"
DEFAULT_RUNNER = f"{REPO}/VITRA/scripts/run_llava_ov2_overfit_qz.sh"
DEFAULT_OUT = f"{REPO}/VITRA/outputs/monkey_cyt_overfit"
QZCLI = os.environ.get(
    "QZCLI",
    "/inspire/ssd/project/robot-reasoning/cengxianchao-240108110052/conda/envs/qzcli/bin/qzcli",
)


def _ensure_login() -> None:
    from qzcli.api import QzAPI
    from qzcli.config import get_cookie, get_credentials, save_cookie

    cookie_data = get_cookie()
    if cookie_data and cookie_data.get("cookie"):
        api = QzAPI()
        try:
            api.list_notebooks_with_cookie(WS_ID, cookie_data["cookie"], page_size=1, status=["RUNNING"])
            print("cookie ok")
            return
        except Exception as exc:
            print(f"cookie invalid: {exc}")

    user, pwd = get_credentials()
    if not user or not pwd:
        user = os.environ.get("QZCLI_USERNAME", "")
        pwd = os.environ.get("QZCLI_PASSWORD", "")
    if not user or not pwd:
        raise SystemExit(
            "缺少认证：请先运行 qzcli login -u <学工号> -p '<密码>'，"
            "或设置 QZCLI_USERNAME / QZCLI_PASSWORD，或 qzcli init。"
        )

    print("logging in via CAS...")
    cookie = QzAPI().login_with_cas(user, pwd)
    save_cookie(cookie, WS_ID)
    print("login ok")


def _find_host() -> str:
    from qzcli.api import QzAPI
    from qzcli.config import get_cookie

    cookie = get_cookie()["cookie"]
    api = QzAPI()
    result = api.list_notebooks_with_cookie(WS_ID, cookie, page_size=100, status=["RUNNING"])
    items = result.get("list") or []
    names = {nb.get("name"): nb for nb in items}
    print(f"[{WS_NAME}] running notebooks: {len(items)}")
    for nb in items:
        cg = (nb.get("logic_compute_group") or {}).get("name", "")
        gpu = (nb.get("quota") or {}).get("gpu_count", "?")
        print(f"  - {nb.get('name')} gpu={gpu} cg={cg}")

    for cand in HOST_CANDIDATES:
        if cand in names:
            return cand
    for nb in items:
        name = (nb.get("name") or "").lower()
        if "monkey" in name and "cyt" in name:
            return nb["name"]
    raise SystemExit(f"在 {WS_NAME} 未找到运行中的 monkey-cyt 开发机")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.environ.get("OUT", DEFAULT_OUT))
    p.add_argument("--runner", default=os.environ.get("RUNNER", DEFAULT_RUNNER))
    p.add_argument(
        "--log",
        default=os.environ.get("MONITOR_LOG", ""),
        help="Remote log file to tail after launch (default: <out>/run.log)",
    )
    return p.parse_args()


def _remote_env() -> str:
    keys = (
        "CFG",
        "EPOCHS",
        "NUM_EVAL",
        "MAX_PRED_VIZ",
        "PRED_INDICES",
        "MAX_SAMPLES",
        "SKIP_PREWARM",
        "CUDA_VISIBLE_DEVICES",
        "EVAL_OUT",
        "VIZ_OUT",
        "CKPT",
    )
    parts = []
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value != "":
            parts.append(f"{key}='{value}'")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    monitor_log = args.log or os.path.join(args.out, "run.log")
    launch_log = os.environ.get("LAUNCH_LOG", monitor_log)
    _ensure_login()
    host = _find_host()
    remote = (
        f"mkdir -p '{args.out}' && "
        f"nohup env OUT='{args.out}' EVAL_OUT='{args.out}/eval' {_remote_env()} bash '{args.runner}' "
        f">> '{launch_log}' 2>&1 & echo started_pid=$!"
    )
    print(f"exec on {host}: {remote}")
    subprocess.run([QZCLI, "exec", host, remote], check=True)
    print(f"launched. monitor: tail -f {monitor_log}")


if __name__ == "__main__":
    main()

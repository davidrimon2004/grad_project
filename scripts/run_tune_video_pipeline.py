#!/usr/bin/env python3
"""
Master Sequential Pipeline for Video Tuning Dataset (Video-ChatGPT)
====================================================================
Runs the complete 3-stage process sequentially without stopping:
  1. DOWNLOAD: Resumes and downloads all split volumes (videochatgpt_tune_2.zip.001..005)
  2. REPAIR: Scans for zeroed blocks or corrupted regions and patches them in-place via HTTP Range requests
  3. EXTRACT: Unpacks split archives -> videochatgpt_tune.zip -> video files (videochatgpt_tune/)

Usage:
  # Run full pipeline with default external drive (E:\\videollava_data):
  python scripts/run_tune_video_pipeline.py

  # Run full pipeline on custom data root:
  python scripts/run_tune_video_pipeline.py --data_root E:\\videollava_data

  # Also delete split volumes after extraction to save ~160 GB:
  python scripts/run_tune_video_pipeline.py --data_root E:\\videollava_data --clean_splits

  # Skip download stage (start directly at repair and extract):
  python scripts/run_tune_video_pipeline.py --skip_download
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path


class Colors:
    HEADER  = "\033[95m"
    OKCYAN  = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL    = "\033[91m"
    ENDC    = "\033[0m"
    BOLD    = "\033[1m"


def log_info(msg: str):
    print(f"{Colors.OKCYAN}[PIPELINE INFO]{Colors.ENDC}    {msg}", flush=True)


def log_success(msg: str):
    print(f"{Colors.OKGREEN}[PIPELINE SUCCESS]{Colors.ENDC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"{Colors.WARNING}[PIPELINE WARN]{Colors.ENDC}    {msg}", flush=True)


def log_err(msg: str):
    print(f"{Colors.FAIL}[PIPELINE ERROR]{Colors.ENDC}   {msg}", flush=True)


def log_banner(title: str):
    bar = "=" * 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{bar}\n  >>> {title}\n{bar}{Colors.ENDC}\n", flush=True)


def run_stage(cmd: list, stage_name: str) -> bool:
    """Run a sub-script command and stream output in real-time."""
    log_banner(f"STARTING STAGE: {stage_name}")
    log_info(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        elapsed_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
        if proc.returncode == 0:
            log_success(f"Stage '{stage_name}' completed successfully in {elapsed_str}!")
            return True
        else:
            log_err(f"Stage '{stage_name}' exited with failure code {proc.returncode} after {elapsed_str}.")
            return False
    except KeyboardInterrupt:
        log_warn(f"Stage '{stage_name}' interrupted by user.")
        raise
    except Exception as exc:
        log_err(f"Stage '{stage_name}' raised exception: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Download, In-Place Repair, and Extraction for Tune Video Dataset sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"E:\videollava_data",
        help="Root directory for storing dataset files (default: E:\\videollava_data)",
    )
    parser.add_argument(
        "--clean_splits",
        action="store_true",
        help="Delete split volumes (videochatgpt_tune_2.zip.001..005) after successful extraction",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep the 160 GB intermediate videochatgpt_tune.zip (default: auto-deleted after unpacking videos)",
    )
    parser.add_argument(
        "--no_aria2",
        action="store_true",
        help="Disable aria2c multi-connection download accelerator",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download stage and go straight to repair and extraction",
    )
    parser.add_argument(
        "--skip_repair",
        action="store_true",
        help="Skip repair stage",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip extraction stage",
    )

    args = parser.parse_args()
    data_root = str(Path(args.data_root).resolve())
    py_exec = sys.executable
    script_dir = Path(__file__).parent.resolve()

    download_script = script_dir / "download_finetune_data.py"
    repair_script = script_dir / "repair_tune_video_zip.py"
    extract_script = script_dir / "extract_tune_videos.py"

    pipeline_start = time.time()
    log_banner("VIDEO-CHATGPT TUNING DATASET AUTOMATED PIPELINE")
    log_info(f"Target Data Root : {data_root}")
    log_info(f"Python Executable: {py_exec}")
    log_info(f"Stages Selected  : Download={'No' if args.skip_download else 'Yes'}, "
             f"Repair={'No' if args.skip_repair else 'Yes'}, "
             f"Extract={'No' if args.skip_extract else 'Yes'}")

    # -------------------------------------------------------------
    # Stage 1: Download
    # -------------------------------------------------------------
    if not args.skip_download:
        dl_cmd = [
            py_exec,
            str(download_script),
            "--data_root", data_root,
            "--components", "video",
        ]
        if args.no_aria2:
            dl_cmd.append("--no_aria2")

        ok = run_stage(dl_cmd, "1/3 Download Video Tuning Dataset")
        if not ok:
            log_err("Pipeline aborted because Download stage failed.")
            sys.exit(1)
    else:
        log_info("Skipping Stage 1 (Download) as requested.")

    # -------------------------------------------------------------
    # Stage 2: Repair Missing Bits / Zero Blocks
    # -------------------------------------------------------------
    if not args.skip_repair:
        repair_cmd = [
            py_exec,
            str(repair_script),
            "--data_root", data_root,
            "--all",
        ]

        ok = run_stage(repair_cmd, "2/3 Scan & Repair Corrupt/Zeroed Regions")
        if not ok:
            log_err("Pipeline aborted because Repair stage found errors.")
            sys.exit(1)
    else:
        log_info("Skipping Stage 2 (Repair) as requested.")

    # -------------------------------------------------------------
    # Stage 3: Extraction
    # -------------------------------------------------------------
    if not args.skip_extract:
        extract_cmd = [
            py_exec,
            str(extract_script),
            "--data_root", data_root,
        ]
        if args.clean_splits:
            extract_cmd.append("--clean_splits")
        if args.keep_intermediate:
            extract_cmd.append("--keep_intermediate")

        ok = run_stage(extract_cmd, "3/3 Extracting Video Clips & Cleanup")
        if not ok:
            log_err("Pipeline aborted because Extraction stage failed.")
            sys.exit(1)
    else:
        log_info("Skipping Stage 3 (Extract) as requested.")

    total_time = time.time() - pipeline_start
    total_str = time.strftime("%Hh %Mm %Ss", time.gmtime(total_time))
    log_banner(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_str}!")
    log_info(f"Videos are available at: {Path(data_root) / 'videochatgpt_tune'}")


if __name__ == "__main__":
    main()

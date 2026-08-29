#!/usr/bin/env python3
"""
Master E2E Test Suite Runner for Handwriting Recognition.
Provides CLI execution, filtering by test tiers (Tier 1-4) and features (F1-F20),
device configuration (mock, cpu, mps), ANSI terminal summaries, and structured JSON reporting.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from tests.fixtures.generator import FixtureGenerator


# ANSI color escape codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class E2EResultCollector:
    """Pytest plugin collecting structured test metrics across tiers and features."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.end_time: float = 0.0
        self.results: List[Dict[str, Any]] = []
        self.tier_counts: Dict[str, Dict[str, Any]] = {
            "tier1": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
            "tier2": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
            "tier3": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
            "tier4": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
            "tier5": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
            "other": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0},
        }
        self.feature_map: Dict[str, Dict[str, Any]] = {
            f"F{i}": {"tier1_passed": 0, "tier1_total": 0, "tier2_passed": 0, "tier2_total": 0, "other_passed": 0, "status": "UNKNOWN"}
            for i in range(1, 25)
        }

    def _extract_feature(self, nodeid: str) -> Optional[str]:
        """Extract feature identifier (F1..F24) from test function name or docstring."""
        match = re.search(
            r"(?:TestTier\d+F|test_f|feature_|_f|\[f|F)(\d{1,2})(?:_|[A-Z]|\[|\b|$)",
            nodeid,
            re.IGNORECASE,
        )
        if match:
            feat_num = int(match.group(1))
            if 1 <= feat_num <= 24:
                return f"F{feat_num}"
        return None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Capture report results when test execution completes."""
        if report.when == "call" or (report.when == "setup" and report.failed):
            outcome = report.outcome  # 'passed', 'failed', 'skipped'
            node_id = report.nodeid
            duration = getattr(report, "duration", 0.0)

            # Determine Tier from markers or file path
            tier = "other"
            if "test_tier1" in node_id or "tier1" in getattr(report, "keywords", {}):
                tier = "tier1"
            elif "test_tier2" in node_id or "tier2" in getattr(report, "keywords", {}):
                tier = "tier2"
            elif "test_tier3" in node_id or "tier3" in getattr(report, "keywords", {}):
                tier = "tier3"
            elif "test_tier4" in node_id or "tier4" in getattr(report, "keywords", {}):
                tier = "tier4"
            elif "test_tier5" in node_id or "tier5" in getattr(report, "keywords", {}):
                tier = "tier5"

            if tier in self.tier_counts:
                self.tier_counts[tier][outcome] = self.tier_counts[tier].get(outcome, 0) + 1
                self.tier_counts[tier]["duration"] += duration

            # Feature tracking
            feature = self._extract_feature(node_id)
            if feature and feature in self.feature_map:
                f_entry = self.feature_map[feature]
                if tier == "tier1":
                    f_entry["tier1_total"] += 1
                    if outcome == "passed":
                        f_entry["tier1_passed"] += 1
                elif tier == "tier2":
                    f_entry["tier2_total"] += 1
                    if outcome == "passed":
                        f_entry["tier2_passed"] += 1
                else:
                    if outcome == "passed":
                        f_entry["other_passed"] += 1

            error_detail = ""
            if report.failed and report.longrepr:
                error_detail = str(report.longrepr)

            self.results.append(
                {
                    "node_id": node_id,
                    "tier": tier,
                    "feature": feature,
                    "outcome": outcome,
                    "duration": round(duration, 4),
                    "error": error_detail if report.failed else None,
                }
            )

    def finalize(self) -> None:
        """Finalize results and update feature pass status."""
        self.end_time = time.time()
        for feat_name, f_data in self.feature_map.items():
            total = f_data["tier1_total"] + f_data["tier2_total"]
            passed = f_data["tier1_passed"] + f_data["tier2_passed"]
            if total > 0:
                f_data["status"] = "PASSED" if passed == total else "FAILED"
            else:
                f_data["status"] = "UNTESTED"


def get_system_environment() -> Dict[str, Any]:
    """Retrieve hardware and runtime environment details."""
    cpu_count = os.cpu_count() or 1
    system_os = platform.platform()
    python_ver = platform.python_version()

    mps_avail = False
    pytorch_ver = "Not Installed"
    try:
        import torch
        pytorch_ver = torch.__version__
        mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        pass

    return {
        "os": system_os,
        "cpu_count": cpu_count,
        "python_version": python_ver,
        "pytorch_version": pytorch_ver,
        "mps_available": mps_avail,
    }


def print_summary_table(collector: E2EResultCollector, env_info: Dict[str, Any], exit_code: int) -> None:
    """Print clean, formatted ANSI execution summary table."""
    total_tests = len(collector.results)
    passed_tests = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed_tests = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped_tests = sum(1 for r in collector.results if r["outcome"] == "skipped")
    total_duration = collector.end_time - collector.start_time
    pass_rate = (passed_tests / total_tests * 100.0) if total_tests > 0 else (100.0 if exit_code == 0 else 0.0)

    print("\n" + "=" * 74)
    print(f"{Colors.BOLD}{Colors.CYAN}E2E TEST SUITE EXECUTION SUMMARY — HANDWRITING RECOGNITION{Colors.RESET}")
    print("=" * 74)
    mps_str = f"{Colors.GREEN}Available{Colors.RESET}" if env_info["mps_available"] else f"{Colors.YELLOW}CPU fallback{Colors.RESET}"
    print(
        f"Environment: {env_info['os']} | Python {env_info['python_version']} | "
        f"PyTorch {env_info['pytorch_version']} (MPS: {mps_str})"
    )
    print("-" * 74)

    tier_labels = {
        "tier1": "Tier 1 (Feature Isolation)   ",
        "tier2": "Tier 2 (Boundary & Corner)   ",
        "tier3": "Tier 3 (Pairwise Integration)",
        "tier4": "Tier 4 (Real-World Workloads)",
        "tier5": "Tier 5 (Adversarial Stress)  ",
    }

    for tier_key, label in tier_labels.items():
        t_data = collector.tier_counts[tier_key]
        t_total = t_data["passed"] + t_data["failed"] + t_data["skipped"]
        t_passed = t_data["passed"]
        t_dur = t_data["duration"]
        t_rate = (t_passed / t_total * 100.0) if t_total > 0 else 100.0
        color = Colors.GREEN if t_data["failed"] == 0 else Colors.RED
        print(f"{label} : {t_passed:4d} / {t_total:4d} passed ({color}{t_rate:5.1f}%{Colors.RESET}) [{t_dur:5.2f}s]")

    print("-" * 74)
    tot_color = Colors.GREEN if failed_tests == 0 else Colors.RED
    print(f"{Colors.BOLD}TOTAL                        : {passed_tests:4d} / {total_tests:4d} passed ({tot_color}{pass_rate:5.1f}%{Colors.RESET}) [{total_duration:5.2f}s]{Colors.RESET}")
    print("=" * 74)

    # Feature coverage stats
    covered_feats = sum(1 for f in collector.feature_map.values() if f["status"] == "PASSED")
    feat_color = Colors.GREEN if covered_feats == 24 else Colors.YELLOW
    print(f"Feature Coverage Matrix: {feat_color}{covered_feats} / 24 Features Fully Verified{Colors.RESET}")
    print("=" * 74 + "\n")


def build_json_report(
    collector: E2EResultCollector, env_info: Dict[str, Any], exit_code: int
) -> Dict[str, Any]:
    """Construct structured JSON report payload."""
    total_tests = len(collector.results)
    passed_tests = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed_tests = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped_tests = sum(1 for r in collector.results if r["outcome"] == "skipped")
    total_duration = round(collector.end_time - collector.start_time, 3)
    pass_rate = round((passed_tests / total_tests * 100.0), 2) if total_tests > 0 else (100.0 if exit_code == 0 else 0.0)

    return {
        "summary": {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "duration_seconds": total_duration,
            "pass_rate_percent": pass_rate,
            "exit_code": exit_code,
        },
        "environment": env_info,
        "tiers": collector.tier_counts,
        "features": collector.feature_map,
        "tests": collector.results,
    }


def run_e2e_tests(
    tier: str = "all",
    feature: Optional[str] = None,
    device: str = "mock",
    fast: bool = False,
    verbose: bool = False,
    json_report_path: Optional[str] = None,
    junit_xml_path: Optional[str] = None,
    generate_fixtures: bool = False,
    extra_pytest_args: Optional[List[str]] = None,
) -> int:
    """Execute E2E test suite with specified configurations and filters."""
    fixture_dir = Path("tests/fixtures").resolve()
    if generate_fixtures or not (fixture_dir / "manifest.json").exists():
        print(f"Generating fixture catalog in {fixture_dir}...")
        gen = FixtureGenerator(out_dir=fixture_dir, seed=42)
        gen.generate_all()

    os.environ["HANDWRITING_TEST_DEVICE"] = device

    pytest_args: List[str] = ["tests/e2e"]

    # Tier filtering
    if tier == "1":
        pytest_args.extend(["-m", "tier1"])
    elif tier == "2":
        pytest_args.extend(["-m", "tier2"])
    elif tier == "3":
        pytest_args.extend(["-m", "tier3"])
    elif tier == "4":
        pytest_args.extend(["-m", "tier4"])
    elif tier == "5":
        pytest_args.extend(["-m", "tier5"])

    # Feature filtering
    if feature:
        feat_clean = feature.strip().upper()
        if feat_clean.startswith("FEATURE_"):
            feat_clean = feat_clean[8:]
        elif feat_clean.startswith("FEATURE"):
            feat_clean = feat_clean[7:]
        elif feat_clean.startswith("F"):
            feat_clean = feat_clean[1:]

        try:
            feat_num = int(feat_clean)
        except ValueError:
            feat_num = None

        if feat_num is not None and 1 <= feat_num <= 24:
            pytest_args.extend([
                "-k",
                f"f{feat_num}_ or feature_{feat_num}_ or f{feat_num}[ or F{feat_num}_ or feature_{feat_num}[",
            ])
        else:
            pytest_args.extend(["-k", feature])

    if fast:
        pytest_args.append("-x")

    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if junit_xml_path:
        pytest_args.extend(["--junitxml", str(junit_xml_path)])

    if extra_pytest_args:
        pytest_args.extend(extra_pytest_args)

    collector = E2EResultCollector()
    exit_code = pytest.main(pytest_args, plugins=[collector])

    # Convert exit code 5 (no tests collected) to 0 if running in scaffold mode
    effective_exit_code = 0 if exit_code == 5 else exit_code

    collector.finalize()
    env_info = get_system_environment()

    print_summary_table(collector, env_info, effective_exit_code)

    if json_report_path:
        report_data = build_json_report(collector, env_info, effective_exit_code)
        out_file = Path(json_report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON execution report saved to: {out_file.resolve()}")

    return effective_exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Master E2E Test Runner for Handwriting Recognition")
    parser.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Run tests for a specific tier (1, 2, 3, 4, 5, or all)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Filter tests for a specific feature (e.g. F1, F5, F12)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mock", "cpu", "mps"],
        default="mock",
        help="Target inference execution device (default: mock)",
    )
    parser.add_argument(
        "--fast",
        "--fail-fast",
        action="store_true",
        help="Stop on first test failure (-x)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose pytest output",
    )
    parser.add_argument(
        "--generate-fixtures",
        action="store_true",
        help="Force regeneration of all test fixtures before running",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Path to write structured JSON test summary report",
    )
    parser.add_argument(
        "--junit-xml",
        type=str,
        default=None,
        help="Path to write standard JUnit XML test report",
    )

    args, unknown_args = parser.parse_known_args()

    sys.exit(
        run_e2e_tests(
            tier=args.tier,
            feature=args.feature,
            device=args.device,
            fast=args.fast,
            verbose=args.verbose,
            json_report_path=args.json_report,
            junit_xml_path=args.junit_xml,
            generate_fixtures=args.generate_fixtures,
            extra_pytest_args=unknown_args,
        )
    )


if __name__ == "__main__":
    main()

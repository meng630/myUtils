#!/usr/bin/env python3

import collections
import csv
import glob
import os
import re
import statistics
import sys
import gzip

rank_re = re.compile(r"^\s*(\d+):\s?(.*)$")
metric_re = re.compile(
    r"^\s*\((ParFor|ParRed|ParScan|Region|REGION)\)\s+"
    r"([-+0-9.eE]+)\s+"
    r"(\d+)\s+"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s*$"
)
number_re = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

case_root = "/pscratch/sd/m/meng/dp_screamxx/mamxx/scream_dpxx_DYCOMSrf01"
log_pattern = os.path.join(case_root, "run_*", "e3sm.log*")
log_files = sorted(glob.glob(log_pattern))
output_tags = ['02', '04', '06', '12']
output_dir = "/pscratch/sd/m/meng/kokkos_tools"

if not log_files:
    print(f"No files matched: {log_pattern}")
    sys.exit(1)

for filename, output_tag in zip(log_files, output_tags):
    
    section = collections.defaultdict(str)
    pending_name = {}
    application_times = {}
    rows = []

    casename = os.path.basename(os.path.dirname(filename))
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rt", encoding="utf-8", errors="replace", ) as stream:
        print(f"Parsing {filename} for case: {casename} ...") 
        for raw_line in stream:
            match = rank_re.match(raw_line)
            if match:
                rank, text = match.groups()
            else:
                rank, text = "unlabeled", raw_line.rstrip()

            key = (filename, rank)
            text = text.strip()

            if text == "Regions:":
                section[key] = "region"
                pending_name.pop(key, None)
                continue

            if text == "Kernels:":
                section[key] = "kernel"
                pending_name.pop(key, None)
                continue

            if text.startswith("- "):
                pending_name[key] = text[2:].strip()
                continue

            match = metric_re.match(text)
            if match and key in pending_name:
                kernel_type, total, calls, average, pct_kernel, pct_program = (
                    match.groups()
                )

                rows.append(
                    {
                        "source": os.path.basename(filename),
                        "rank": rank,
                        "section": section[key],
                        "kernel_type": kernel_type,
                        "kernel_name": pending_name.pop(key),
                        "total_seconds": float(total),
                        "call_count": int(calls),
                        "seconds_per_call": float(average),
                        "pct_total_time_in_kernels": float(pct_kernel),
                        "pct_total_program_time": float(pct_program),
                    }
                )
                continue

            if text.startswith("Total Execution Time"):
                numbers = number_re.findall(text)
                if numbers:
                    application_times[key] = float(numbers[-1])

    with open(os.path.join(output_dir, f"kernel_timer_by_rank_{output_tag}.csv"), "w", newline="") as stream:
        fields = [
            "source",
            "rank",
            "section",
            "kernel_type",
            "kernel_name",
            "total_seconds",
            "call_count",
            "seconds_per_call",
            "pct_total_time_in_kernels",
            "pct_total_program_time",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    kernel_rows = [row for row in rows if row["section"] == "kernel"]
    total_kernel_rank_seconds = sum(row["total_seconds"] for row in kernel_rows)
    total_program_rank_seconds = sum(application_times.values())

    groups = collections.defaultdict(lambda: {"times": [], "calls": 0, "ranks": set()})

    for row in kernel_rows:
        key = (row["kernel_type"], row["kernel_name"])
        groups[key]["times"].append(row["total_seconds"])
        groups[key]["calls"] += row["call_count"]
        groups[key]["ranks"].add((row["source"], row["rank"]))

    aggregated = []

    for (kernel_type, kernel_name), values in groups.items():
        times = values["times"]
        summed_time = sum(times)

        aggregated.append(
            {
                "kernel_type": kernel_type,
                "kernel_name": kernel_name,
                "ranks_present": len(values["ranks"]),
                "sum_rank_seconds": summed_time,
                "mean_rank_seconds": statistics.mean(times),
                "min_rank_seconds": min(times),
                "max_rank_seconds": max(times),
                "total_calls": values["calls"],
                "seconds_per_call": (
                    summed_time / values["calls"] if values["calls"] else 0.0
                ),
                "pct_all_kernel_rank_time": (
                    100.0 * summed_time / total_kernel_rank_seconds
                    if total_kernel_rank_seconds
                    else 0.0
                ),
                "pct_all_program_rank_time": (
                    100.0 * summed_time / total_program_rank_seconds
                    if total_program_rank_seconds
                    else 0.0
                ),
            }
        )

    aggregated.sort(key=lambda row: row["sum_rank_seconds"], reverse=True)

    with open(os.path.join(output_dir, f"kernel_timer_aggregated_{output_tag}.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregated[0].keys())
        writer.writeheader()
        writer.writerows(aggregated)

    print(f"Parsed {len(rows)} timing records")
    print(f"Found reports from {len(application_times)} rank executions")
    print(f"Created kernel_timer_by_rank_{output_tag}.csv")
    print(f"Created kernel_timer_aggregated_{output_tag}.csv")
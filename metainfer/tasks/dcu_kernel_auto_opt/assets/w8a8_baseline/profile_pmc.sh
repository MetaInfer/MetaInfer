#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: profile_pmc.sh HARNESS SOURCE M N K OUTPUT_DIR" >&2
    exit 2
fi

harness=$1
source_dir=$2
m=$3
n=$4
k=$5
output_dir=$6

run_profile() {
    mode=$1
    name=$2
    csv_path="$output_dir/$name.csv"
    mkdir -p "$output_dir/data-$name"
    /opt/dtk/bin/hipprof \
        "$mode" \
        --pmc-type 3 \
        --flush-interval 1000 \
        --exit-cleanup \
        -o "$output_dir/$name" \
        -d "$output_dir/data-$name" \
        python3 "$harness" \
        --source "$source_dir" \
        --m "$m" \
        --n "$n" \
        --k "$k" \
        --warmups 0 \
        --samples 1 \
        --replays-per-sample 1 \
        --profile-only

    # pmc-read/pmc-write use replay and their launcher may return before the
    # replay child flushes the CSV. Serialize the counter groups and make a
    # missing result a hard profiling failure.
    deadline=$((SECONDS + 300))
    while [ ! -s "$csv_path" ]; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "hipprof did not produce $csv_path within 300 seconds" >&2
            return 1
        fi
        sleep 1
    done
}

# General resources/instructions and memory traffic are separate hipprof
# counter groups. Keep all three raw CSV files so the control plane can apply
# DTK's documented request-size formulas without trusting Agent arithmetic.
run_profile --pmc pmc
run_profile --pmc-read pmc-read
run_profile --pmc-write pmc-write

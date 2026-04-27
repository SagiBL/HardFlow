#!/bin/bash
# Run the text-guided image-editing experiment under five prompts.
set -euo pipefail

run_experiment() {
	local method="$1"
	echo "Starting experiment with method: $method"

	local start_time end_time elapsed hours remainder minutes seconds
	start_time=$(date +%s)
	python main.py --folder experiment --method "$method"
	end_time=$(date +%s)

	elapsed=$((end_time - start_time))
	hours=$((elapsed / 3600))
	remainder=$((elapsed % 3600))
	minutes=$((remainder / 60))
	seconds=$((remainder % 60))

	printf "Total time for %s: %d hours, %d minutes, %d seconds\n\n" \
		"$method" "$hours" "$minutes" "$seconds"
}

echo "Running image-editing experiments..."
run_experiment "gradient_guidance"
run_experiment "oc_flow"
run_experiment "projection_relaxed"
run_experiment "hardflow"
echo "All experiments completed."

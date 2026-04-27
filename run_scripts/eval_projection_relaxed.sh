#!/bin/bash
# Evaluate the Projection-Relaxed baselines on the 1-D Burgers PDE control task.
# Runs two configs:
#   Projection-Relaxed
#   Projection-Relaxed + Gradient Guidance
start_time=$(date +%s)

flow_exp_name="4e5steps"
flow_cp=40

ode_t_steps=10
n_test_samples=50

constraint="safety_score"
safety_threshold_margin=0.08
safety_constraint_profile="quadratic"

value_objective="control_energy"
value_objective_scale=0.1
value_constraint_scale=0.0

run_one () {
    local g_steps="$1"
    local label="projection_relaxed"
    if [ "$g_steps" -gt 0 ]; then
        label="${label}_gradient_guidance"
    fi
    local eval_exp_name="4e5steps_${label}_${ode_t_steps}steps"

    echo "=== Running ${label} on Burgers (n_test_samples=${n_test_samples}) ==="
    echo "Results will be saved to: ./logs/burgers/eval/${eval_exp_name}/"

    python run/eval.py \
        --log_folder ./logs \
        --flow_exp_name "$flow_exp_name" \
        --eval_exp_name "$eval_exp_name" \
        --flow_cp $flow_cp \
        --data_path datasets/burgers \
        --n_test_samples $n_test_samples \
        --ode_t_steps $ode_t_steps \
        --constraint "$constraint" \
        --safety_threshold_margin $safety_threshold_margin \
        --safety_constraint_profile "$safety_constraint_profile" \
        --guidance_method projection_relaxed \
        --value_objective "$value_objective" \
        --value_objective_scale $value_objective_scale \
        --value_constraint_scale $value_constraint_scale \
        --projection_gradient_steps "$g_steps" \
        --device "cuda:0" \
        --seed 42
}

run_one 0    # Projection-Relaxed
run_one 20   # Projection-Relaxed + Gradient Guidance

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

#!/bin/bash
# Evaluate the Projection-Relaxed baselines on maze2d-large-v1.
# Runs two configs:
#   Projection-Relaxed
#   Projection-Relaxed + Gradient Guidance
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="maze2d-large-v1"
state_dim=4
action_dim=2

flow_type="cfm"
horizon=384
flow_cp=20
ode_t_steps=10

random_repeat=1
controller="pd"

warmstart_batch=1
value_objective="distance"
value_objective_scale=2.0
value_constraint_scale=0.0

constraint="ellipses"
ellipse_margin=0.4
solver_print_level=5

run_one () {
    local g_steps="$1"
    local label="projection_relaxed"
    if [ "$g_steps" -gt 0 ]; then
        label="${label}_gradient_guidance"
    fi
    local exp_name="H${horizon}_1e6steps_${label}_${ode_t_steps}steps"

    echo "=== Running ${label} on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps} ==="

    python run/eval.py \
        --device cpu \
        --seed 0 \
        --random_repeat "$random_repeat" \
        --exp_name "$exp_name" \
        --env "$env" \
        --state_dim "$state_dim" \
        --action_dim "$action_dim" \
        --horizon "$horizon" \
        --flow_exp_name "H${horizon}_1e6steps" \
        --flow_cp "$flow_cp" \
        --flow_matching_type "$flow_type" \
        --ode_t_steps "$ode_t_steps" \
        --warmstart_batch "$warmstart_batch" \
        --value_objective "$value_objective" \
        --value_objective_scale "$value_objective_scale" \
        --value_constraint_scale "$value_constraint_scale" \
        --solver_print_level "$solver_print_level" \
        --constraint "$constraint" \
        --ellipse_margin "$ellipse_margin" \
        --controller "$controller" \
        --projection_gradient_steps "$g_steps" \
        --no-render \
        --fixed_start \
        --guidance_method projection_relaxed
}

run_one 0   # Projection-Relaxed
run_one 5   # Projection-Relaxed + Gradient Guidance

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

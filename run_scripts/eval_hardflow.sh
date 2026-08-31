#!/bin/bash
# Evaluate HardFlow on maze2d-large-v1.
# HardFlow steers the sampler so the planned trajectory avoids all obstacles
# and reaches the goal as fast as possible.
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

# warmstart (one initial trajectory drawn from the nominal sampler) is shared
# with all baselines that need a starting iterate
warmstart_batch=1
value_objective="distance"
value_objective_scale=2.0
value_constraint_scale=0.1

# HardFlow surrogate optimization
constraint="ellipses_and_dynamics"
ellipse_margin=0.4
cost="distance"
hardflow_cost_scale=800.0
hardflow_activation="late"  # skip control in the first half of sampling steps
solver_print_level=5

exp_name="H${horizon}_1e6steps_hardflow_${ode_t_steps}steps"

echo "=== Running HardFlow on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps} ==="

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
	--cost "$cost" \
	--ellipse_margin "$ellipse_margin" \
	--hardflow_cost_scale "$hardflow_cost_scale" \
	--hardflow_activation "$hardflow_activation" \
	--controller "$controller" \
	--no-render \
	--fixed_start \
	--guidance_method hardflow

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

#!/bin/bash
# Evaluate the Gradient Guidance baseline on maze2d-large-v1.
# Soft-constrained gradient is computed on the value model (distance + penalty)
# and applied to intermediate samples during ODE integration.
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
batch_size=1

constraint="ellipses"
ellipse_margin=0.4
value_objective="distance"
value_objective_scale=2.0
value_constraint_scale=0.1

exp_name="H${horizon}_1e6steps_gradient_guidance_${ode_t_steps}steps"

echo "=== Running Gradient Guidance on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps} ==="

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
	--constraint "$constraint" \
	--ellipse_margin "$ellipse_margin" \
	--batch_size "$batch_size" \
	--value_objective "$value_objective" \
	--value_objective_scale "$value_objective_scale" \
	--value_constraint_scale "$value_constraint_scale" \
	--controller "$controller" \
	--no-render \
	--fixed_start \
	--guidance_method gradient_guidance

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

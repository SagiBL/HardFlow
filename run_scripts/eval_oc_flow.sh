#!/bin/bash
# Evaluate the OC-Flow baseline on maze2d-large-v1.
# OC-Flow uses indirect optimal control (PMP) to perturb the velocity
# field and minimize a soft cost; constraints are not strictly enforced.
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="maze2d-large-v1"
state_dim=4
action_dim=2

flow_type="cfm"
horizon=384
flow_cp=20
ode_t_steps=11

random_repeat=20
controller="pd"

# warmstart trajectory drawn from the nominal sampler
warmstart_batch=1

# OC-Flow combines distance objective with a constraint penalty into the
# value model that supplies the indirect gradient
value_objective="distance"
value_objective_scale=2.0
value_constraint_scale=0.1
constraint="ellipses_and_dynamics"
ellipse_margin=0.4

# OC-Flow indirect gradient settings
oc_flow_lr=0.1
oc_flow_steps=20

exp_name="H${horizon}_1e6steps_oc_flow_${ode_t_steps}steps"

echo "=== Running OC-Flow on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps} ==="

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
	--constraint "$constraint" \
	--ellipse_margin "$ellipse_margin" \
	--oc_flow_lr "$oc_flow_lr" \
	--oc_flow_steps "$oc_flow_steps" \
	--controller "$controller" \
	--no-render \
	--fixed_start \
	--guidance_method oc_flow

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

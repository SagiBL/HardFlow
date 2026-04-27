#!/bin/bash
# Evaluate OC-Flow on the 1-D Burgers PDE control task.
start_time=$(date +%s)

flow_exp_name="4e5steps"
flow_cp=40

ode_t_steps=10
n_test_samples=50

constraint="safety_score"
cost="control_energy"
safety_threshold_margin=0.08
safety_constraint_profile="quadratic"

value_objective="control_energy"
value_objective_scale=0.1
value_constraint_scale=1.0

eval_exp_name="4e5steps_oc_flow_${ode_t_steps}steps"

echo "=== Running OC-Flow on Burgers (n_test_samples=${n_test_samples}) ==="

python run/eval.py \
	--log_folder ./logs \
	--flow_exp_name "$flow_exp_name" \
	--eval_exp_name "$eval_exp_name" \
	--flow_cp $flow_cp \
	--data_path datasets/burgers \
	--n_test_samples $n_test_samples \
	--ode_t_steps $ode_t_steps \
	--constraint "$constraint" \
	--cost "$cost" \
	--safety_threshold_margin $safety_threshold_margin \
	--safety_constraint_profile "$safety_constraint_profile" \
	--guidance_method "oc_flow" \
	--value_objective "$value_objective" \
	--value_objective_scale $value_objective_scale \
	--value_constraint_scale $value_constraint_scale \
	--device "cuda:0" \
	--seed 42

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

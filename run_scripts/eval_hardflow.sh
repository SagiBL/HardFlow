#!/bin/bash
# Evaluate HardFlow on the 1-D Burgers PDE control task.
start_time=$(date +%s)

flow_exp_name="4e5steps"
flow_cp=40

ode_t_steps=10
n_test_samples=50

constraint="safety_score"
cost="control_energy"
hardflow_cost_scale=1.0
safety_threshold_margin=0.08
safety_constraint_profile="quadratic"

value_objective="control_energy"
value_objective_scale=0.1
value_constraint_scale=1.0

hardflow_activation="late"

eval_exp_name="4e5steps_hardflow_${ode_t_steps}steps"

echo "=== Running HardFlow on Burgers (n_test_samples=${n_test_samples}) ==="

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
	--hardflow_cost_scale $hardflow_cost_scale \
	--hardflow_activation "$hardflow_activation" \
	--safety_threshold_margin $safety_threshold_margin \
	--safety_constraint_profile "$safety_constraint_profile" \
	--guidance_method "hardflow" \
	--value_objective "$value_objective" \
	--value_objective_scale $value_objective_scale \
	--value_constraint_scale $value_constraint_scale \
	--device "cuda:0" \
	--seed 42

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))

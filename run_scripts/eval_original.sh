start_time=$(date +%s)

echo "=== Evaluating ==="

flow_exp_name="4e5steps"
flow_cp=40

ode_t_steps=10
n_test_samples=50

eval_exp_name="4e5steps_original_${ode_t_steps}steps"

CHECKPOINT_PATH="./logs/burgers/flow/${flow_exp_name}/model_ema_${flow_cp}.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
	echo "Warning: EMA checkpoint not found at $CHECKPOINT_PATH"
	CHECKPOINT_PATH="./logs/burgers/flow/${flow_exp_name}/model_${flow_cp}.pth"
	if [ ! -f "$CHECKPOINT_PATH" ]; then
		echo "Error: No checkpoint found for cp=${flow_cp}"
		echo "Looking for: $CHECKPOINT_PATH"
		exit 1
	fi
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Results will be saved to: ./logs/burgers/eval/${eval_exp_name}/"

python run/eval.py \
	--log_folder ./logs \
	--flow_exp_name "$flow_exp_name" \
	--eval_exp_name "$eval_exp_name" \
	--flow_cp $flow_cp \
	--data_path datasets/burgers \
	--n_test_samples $n_test_samples \
	--ode_t_steps $ode_t_steps \
	--guidance_method "original" \
	--device "cuda:0" \
	--seed 42

end_time=$(date +%s)
elapsed=$((end_time - start_time))

hours=$((elapsed / 3600))
remainder=$((elapsed % 3600))
minutes=$((remainder / 60))
seconds=$((remainder % 60))

printf "Total runtime: %d hours, %d minutes, %d seconds\n" \
	"$hours" "$minutes" "$seconds"

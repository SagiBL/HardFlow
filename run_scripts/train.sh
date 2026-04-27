start_time=$(date +%s)

echo "=== Training ==="

exp_name="4e5steps"

echo "Experiment: $exp_name"
echo "Models will be saved to: ./logs/burgers/flow/${exp_name}/"

python run/train.py \
	--gpu_id 0 \
	--data_path datasets/burgers \
	--use_conditioning True \
	--n_train_steps 400001 \
	--batch_size 16 \
	--learning_rate 1e-5 \
	--ema_decay 0.995 \
	--grad_clip 1.0 \
	--save_freq 10000 \
	--log_folder ./logs \
	--exp_name "$exp_name" \
	--seed 42

end_time=$(date +%s)
elapsed=$((end_time - start_time))

hours=$((elapsed / 3600))
remainder=$((elapsed % 3600))
minutes=$((remainder / 60))
seconds=$((remainder % 60))

printf "Total runtime: %d hours, %d minutes, %d seconds\n" \
	"$hours" "$minutes" "$seconds"

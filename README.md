# 1. Train GPT and collect activations
python train_gpt.py \
  --data_path qm9.csv \
  --output_dir experiments/exp_001 \
  --hook_point blocks.0.hook_resid_post \
  --epochs 20 \
  --batch_size 512

# 2. Train SAE
python train_sae.py \
  --input_dir experiments/exp_001 \
  --epochs 50 \
  --batch_size 1024 \
  --l1_coeff 1e-3

# 3. Evaluate on all splits
python eval.py \
  --exp_dir experiments/exp_001 \
  --generate_dashboard

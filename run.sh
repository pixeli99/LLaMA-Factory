llamafactory-cli train \
  --stage sft \
  --model_name_or_path apple/DiffuCoder-7B-cpGRPO \
  --dataset synthetic_sft \
  --dataset_dir /zju_0038/pengxiang/LLaMA-Factory/data \
  --template qwen \
  --cutoff_len 4096 \
  --finetuning_type full \
  --use_dlm true
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir ../saved_models/tevatron_llama3.1_8b_RARe_30_70 \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora \
  --lora_r 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --query_prefix "" \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 1024 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 4 \
  --ddp_find_unused_parameters false \
  --n_ic_examples 5


# deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
#   --deepspeed deepspeed/ds_zero3_config.json \
#   --output_dir ../saved_models/tevatron_llama3.1_8b_repllama \
#   --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --lora \
#   --lora_r 32 \
#   --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
#   --save_steps 500 \
#   --dataset_name Tevatron/msmarco-passage-aug \
#   --query_prefix "Query: " \
#   --passage_prefix "Passage: " \
#   --bf16 \
#   --pooling eos \
#   --append_eos_token \
#   --normalize \
#   --temperature 0.01 \
#   --per_device_train_batch_size 8 \
#   --gradient_checkpointing \
#   --train_group_size 16 \
#   --learning_rate 1e-4 \
#   --query_max_len 1024 \
#   --passage_max_len 196 \
#   --num_train_epochs 1 \
#   --logging_steps 10 \
#   --overwrite_output_dir \
#   --warmup_steps 100 \
#   --gradient_accumulation_steps 4 \
#   --ddp_find_unused_parameters false \
#   --n_ic_examples 0
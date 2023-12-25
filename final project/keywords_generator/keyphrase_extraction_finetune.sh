python keyphrase_extraction_finetune.py \
    --model_name_or_path "distilbert-base-uncased" \
    --output_dir "distilbert_keyphrase_extraction" \
    --max_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \

# models:
# distilbert-base-uncased   (learning_rate 1e-4, effective_batch_size 64)
# bloomberg/KBIR  (learning_rate 5e-6, effective_batch_size 64)
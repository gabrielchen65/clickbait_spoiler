python3 run_summarization.py \
    --model_name_or_path Einmalumdiewelt/T5-Base_GNAD \
    --do_train \
    --do_eval \
    --train_file ./datasets/task2/train_4summary_multi-concat.json \
    --validation_file ./datasets/task2/val_4summary_multi-concat.json \
    --test_file ./datasets/task2/val_4summary_multi-concat.json \
    --dataset_config "3.0.0" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --save_steps 1500 \
    --output_dir ./tmp/Einmalumdiewelt-T5-Base_GNAD/concat/ \
    --max_target_length 70 \
    --save_total_limit 2 \
    --save_strategy 'no' \
    --load_best_model_at_end False \
    --source_prefix "summarize: "

    #    --source_prefix "summarize: " \ only t5 model
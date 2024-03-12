python3 run_summarization.py \
    --model_name_or_path Einmalumdiewelt/T5-Base_GNAD \
    --do_train \
    --do_eval \
    --text_column article \
    --summary_column highlights \
    --train_file ./datasets/pilot_test.json \
    --validation_file ./datasets/pilot_test.json \
    --dataset_config "3.0.0" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --num_train_epochs 2 \
    --output_dir ./tmp/Einmalumdiewelt-T5-Base_GNAD/concat/ \
    --max_target_length 40 \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --source_prefix "summarize: " \
    --evaluation_strategy epoch

    #    --source_prefix "summarize: " \ only t5 model
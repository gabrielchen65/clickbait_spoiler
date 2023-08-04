python3 run_qa.py \
  --model_name_or_path ./tmp/deepset-roberta-base-squad2-768/checkpoint-12000/ \
  --validation_file ./datasets/val_sum_t5-base_tuned_multi.json \
  --do_eval \
  --evaluation_metric meteor \
  --per_device_train_batch_size 12 \
  --max_seq_length 768 \
  --doc_stride 128 \
  --output_dir ./tmp/test/ \
  #--version_2_with_negative

#./tmp/eval_sum/val_sum_t5-base_tuned-no-train-qa/output_30epoches_distilbert-base-uncased-distilled-squad/checkpoint-22500/ \
# distilbert-base-uncased-distilled-squad
# elozano/bert-base-cased-clickbait-news
# LeonardBongard/deepset_deberta-v3-base-squad2-clickbait-spoiler-generation
#  eval_exact_match        =       34.5
#  eval_f1                 =    50.0079
#  eval_runtime            = 0:06:39.80
#  eval_samples            =       1132
#  eval_samples_per_second =      2.831
#  eval_steps_per_second   =      0.355
# tmp/output_30epoches_distilbert-base-uncased-distilled-squad/checkpoint-22500/

# my best
# deepset-roberta-base-squad2-256stride

python3 run_qa.py \
  --model_name_or_path deepset/roberta-base-squad2 \
  --train_file ./datasets/pilot_test.json \
  --validation_file ./datasets/pilot_test.json \
  --test_file ./datasets/pilot_test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 386 \
  --doc_stride 128 \
  --output_dir ./tmp/pilot_test \
  --save_strategy epoch \
  --training_w_peft True \
  --evaluation_strategy epoch
  #--version_2_with_negative


# distilbert-base-uncased-distilled-squad
  # --model_name_or_path ./tmp/output_30epoches_distilbert-base-uncased-distilled-squad/checkpoint-22500/ \
  # --train_file ./datasets/train.json \
  # --validation_file ./datasets/val.json \
  # --test_file ./datasets/val.json \
  # --do_train \
  # --do_eval \
  # --per_device_train_batch_size 12 \
  # --learning_rate 3e-5 \
  # --num_train_epochs 30 \
  # --max_seq_length 384 \
  # --doc_stride 128 \
  # --output_dir ./tmp/output_v2/

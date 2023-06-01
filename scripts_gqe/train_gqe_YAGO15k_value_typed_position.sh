CUDA_VISIBLE_DEVICES=6 python ../model/train.py \
    --train_query_dir ../input_files/YAGO15K_train_queries.pkl \
    --valid_query_dir ../input_files/YAGO15K_valid_queries.pkl \
    --test_query_dir ../input_files/YAGO15K_test_queries.pkl \
    --data_name YAGO15K \
    --model gqe \
    --batch_size 1024 \
    --typed \
    --log_steps 60000 \
    --numeral_encoder positional


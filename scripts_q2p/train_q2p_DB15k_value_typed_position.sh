CUDA_VISIBLE_DEVICES=3 python ../model/train.py \
    --train_query_dir ../input_files/DB15K_train_queries.pkl \
    --valid_query_dir ../input_files/DB15K_valid_queries.pkl \
    --test_query_dir ../input_files/DB15K_test_queries.pkl \
    --data_name DB15K \
    --model q2p \
    --batch_size 1024 \
    --typed \
    --log_steps 120000 \
    --numeral_encoder positional


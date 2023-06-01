CUDA_VISIBLE_DEVICES=5 python ../model/train_baseline.py \
    --train_query_dir ../input_files/FB15K_train_queries.pkl \
    --valid_query_dir ../input_files/FB15K_valid_queries.pkl \
    --test_query_dir ../input_files/FB15K_test_queries.pkl \
    --data_name FB15K \
    --model gqe \
    --batch_size 1024 \
    --log_steps 60000 
   
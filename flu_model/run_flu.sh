#!/bin/bash


python ./main_flu.py --data_path=/home/hyungjun/jupyter/KT_covid19/data/ \
                     --result_path=/home/hyungjun/jupyter/KT_covid19/results/ \
                     --device=cuda:0 \
                     --batch_size=1 \
                     --num_epochs=200 \
                     --learning_rate=5e-3 \
                     --train_days=296 \
                     --delay_days=0 \
                     --test_days=148 \
                     --lstm_sequence_length=7 \
                     --lstm_num_layers=1 \
                     --lstm_input_size=50 \
                     --lstm_hidden_size=100 \
                     --graph_conv_feature_dim_user=15 \
                     --loss=mse \
                     --data_normalize \
                     --num_dong=25 \
                     --concat_fc \
                     --deep_graph

echo "finished"
#!/bin/bash


python ./main_covid.py --data_path=/home/hyungjun/jupyter/KT_covid19/data/ \
                       --result_path=/home/hyungjun/jupyter/KT_covid19/results/ \
                       --task=real_feb \
                       --loss=mse \
                       --data_normalize \
                       --num_dong=139 \
                       --num_epochs=300 \
                       --learning_rate=1 \
                       --train_days=21 \
                       --delay_days=0 \
                       --test_days=7 \
                       --lstm_sequence_length=1 \
                       --lstm_num_layers=1 \
                       --lstm_input_size=50 \
                       --lstm_hidden_size=100 \
                       --graph_conv_feature_dim_user=20 \
                       --graph_conv_feature_dim_infect=20 \
                       --concat_fc \
                       --deep_graph

echo "finished"
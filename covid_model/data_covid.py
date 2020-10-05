import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import networkx as nx
import torch_geometric
import os


def data_loader_covid(args):
    data_filename = os.path.join('covid_norm{}_d{}-{}-{}_s{}'.format(args.data_normalize, args.train_days, args.delay_days, args.test_days, args.lstm_sequence_length))
    
    print('== data import ==')
    # Dataset import 
    edge_daegu_infect = np.load(args.data_path + 'raw_data/covid/edge_daegu_infect_half_feb.npy', allow_pickle=True)
    edge_daegu_user = np.load(args.data_path + 'raw_data/covid/edge_daegu_user_half_feb.npy', allow_pickle=True) # 2월
    node_daegu_infect = torch.from_numpy(np.load(args.data_path + 'raw_data/covid/node_daegu_infect_half_feb.npy', allow_pickle=True))
    node_daegu_user = torch.from_numpy(np.load(args.data_path + 'raw_data/covid/node_daegu_user_half_feb.npy', allow_pickle=True)) # 2월
    
    # order : user, infect , user_full, infect_full
    min_max_values = [torch.min(node_daegu_user).item(), torch.max(node_daegu_user).item(), 
                      np.min(edge_daegu_user[:,4]), np.max(edge_daegu_user[:,4]), 
                      torch.min(node_daegu_infect).item(), torch.max(node_daegu_infect).item(),
                      np.min(edge_daegu_user[:,4]), np.max(edge_daegu_user[:,4])]

    if not os.path.exists(os.path.join(args.data_path + 'train_'+ data_filename)):
        # hyperparameters
        LSTM_SEQUENCE_LENGTH = args.lstm_sequence_length
        TRAIN_DAYS = args.train_days
        DELAY_DAYS = args.delay_days
        TEST_DAYS = args.test_days
        DEVICE = args.device
        NUM_EPOCHS = args.num_epochs
        LEARNING_RATE = args.learning_rate
        NUM_DONG = args.num_dong

        # normalize (node, edge)
        if args.data_normalize:
            node_user = (node_daegu_user.reshape([-1, NUM_DONG])-torch.min(node_daegu_user))/(torch.max(node_daegu_user)-torch.min(node_daegu_user))
            node_infect = (node_daegu_infect.reshape([-1, NUM_DONG])-torch.min(node_daegu_infect))/ \
                          (torch.max(node_daegu_infect)-torch.min(node_daegu_infect))
            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3],
                                  (edge_daegu_user[:,4]-np.min(edge_daegu_user[:,4]))/ \
                                  (np.max(edge_daegu_user[:,4])-np.min(edge_daegu_user[:,4]))]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3],
                                    (edge_daegu_infect[:,4]-np.min(edge_daegu_infect[:,4]))/ \
                                    (np.max(edge_daegu_infect[:,4])-np.min(edge_daegu_infect[:,4]))]).transpose()
   
        else:
            node_user = node_daegu_user.reshape([-1, NUM_DONG])
            node_infect = node_daegu_infect.reshape([-1, NUM_DONG])
            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3], 
                                  edge_daegu_user[:,4]]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3], 
                                    edge_daegu_infect[:,4]]).transpose()
        
        # node ready
        print('=== node_ready ===')
        train_infect_node_list, train_user_node_list = [], []
        test_infect_node_list, test_user_node_list = [], []

        for i in range(48*TRAIN_DAYS):
            train_infect_node_list.append(node_infect[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            train_user_node_list.append(node_user[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        for i in range(48*TRAIN_DAYS, 48*(TRAIN_DAYS + TEST_DAYS)):
            test_infect_node_list.append(node_infect[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            test_user_node_list.append(node_user[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        # edge ready
        print('=== edge_ready ===')
        ymd_time_lst = np.sort(np.unique(edge_user[:,0]))
        dong_lst = np.sort(np.unique(edge_daegu_user[:,2]))

        user_edge_inputs, infect_edge_inputs = [], []

        # edge generation using network X
        for ymd_time in tqdm(ymd_time_lst):
            g = nx.DiGraph()
            g.add_edges_from(edge_user[(edge_user[:,0]==ymd_time)][:,1:3])
            user_edge_list = torch_geometric.utils.from_networkx(g)
            user_edge_weight = edge_user[(edge_user[:,0]==ymd_time)][:,3]
            user_edge_inputs.append([user_edge_list, user_edge_weight])

        for ymd_time in tqdm(ymd_time_lst):
            g = nx.DiGraph()
            g.add_edges_from(edge_infect[(edge_infect[:,0]==ymd_time)][:,1:3])
            infect_edge_list = torch_geometric.utils.from_networkx(g)
            infect_edge_weight = edge_infect[(edge_infect[:,0]==ymd_time)][:,3]
            infect_edge_inputs.append([infect_edge_list, infect_edge_weight])

        train_user_edge_inputs, train_infect_edge_inputs = [], []
        test_user_edge_inputs, test_infect_edge_inputs = [], []

        for i in range(48 * TRAIN_DAYS):
            train_user_edge_inputs.append(user_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            train_infect_edge_inputs.append(infect_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        for i in range(48 * TRAIN_DAYS, 48 * (TRAIN_DAYS + TEST_DAYS)):
            test_user_edge_inputs.append(user_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            test_infect_edge_inputs.append(infect_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        
        # temporal signal (20.02.01 ~ 20.02.29)
        print('== temporal_signal ready ==')
        weekday_list = ['sat', 'sun', 'mon', 'tue','wed','thu','fri']
        
        temp_sig = torch.zeros([29*48, 7 + 48 + 1 + 1]) 
        for i in range(4): # 4 weeks
            for j, weekday in enumerate(weekday_list):
                for time in range(48):
                    temp_sig[(48*7)*i + 48*j + time][j] = 1
                    temp_sig[(48*7)*i + 48*j + time][7 + time] = 1
                    if j == 6: # 금요일
                        temp_sig[(48*7)*i + 48*j + time][7+48+1] = 1 # before holiday
                    elif j == 0: # 토요일
                        temp_sig[(48*7) + 48*j + time][7+48+1] = 1 # before holiday
                        temp_sig[(48*7) + 48*j + time][7+48] = 1 # holiday
                    elif j == 1: # 일요일
                        temp_sig[(48*7) + 48*j + time][7+48] = 1 # holiday

        # the last day(29th)
        for time in range(48):
            temp_sig[28*48 + time][1] = 1
            temp_sig[28*48 + time][7 + time] = 1
            temp_sig[28*48 + time][7+48+1] = 1 # before holiday
            temp_sig[28*48 + time][7+48] = 1 # holiday

        train_temp_sig_inputs, test_temp_sig_inputs = [], []
        for i in range(48 * TRAIN_DAYS):
            train_temp_sig_inputs.append(temp_sig[i:i+48*LSTM_SEQUENCE_LENGTH])
        for i in range(48 * TRAIN_DAYS, 48 * (TRAIN_DAYS + TEST_DAYS)):
            test_temp_sig_inputs.append(temp_sig[i:i+48*LSTM_SEQUENCE_LENGTH])

        # input concatenate (graph user, graph infect, temp guide)
        print("=== concat ===")
        train_inputs = []
        for i in tqdm(range(48*TRAIN_DAYS)):
            train_g_user_inputs, train_g_infect_inputs = [], []
            for j in range(48*LSTM_SEQUENCE_LENGTH):
                train_g_user_inputs.append([train_user_node_list[i][j], train_user_edge_inputs[i][j][0].edge_index, 
                                                    torch.tensor(train_user_edge_inputs[i][j][1].astype(float))])
                train_g_infect_inputs.append([train_infect_node_list[i][j], train_infect_edge_inputs[i][j][0].edge_index, 
                                                      torch.tensor(train_infect_edge_inputs[i][j][1].astype(float))])
            train_inputs.append([train_g_user_inputs, train_g_infect_inputs, train_temp_sig_inputs[i]])

        test_inputs = []
        for i in tqdm(range(48*TEST_DAYS)):
            test_g_user_inputs, test_g_infect_inputs = [], []
            for j in range(48*LSTM_SEQUENCE_LENGTH):
                test_g_user_inputs.append([test_user_node_list[i][j], test_user_edge_inputs[i][j][0].edge_index, 
                                                   torch.tensor(test_user_edge_inputs[i][j][1].astype(float))])
                test_g_infect_inputs.append([test_infect_node_list[i][j], test_infect_edge_inputs[i][j][0].edge_index, 
                                                     torch.tensor(test_infect_edge_inputs[i][j][1].astype(float))])
            test_inputs.append([test_g_user_inputs, test_g_infect_inputs, test_temp_sig_inputs[i]])

        ###################
        ###### label ######
        
        print('== label ready ==')
        # user, infect
        # sty label
        node_daegu_user_reshape = node_daegu_user.reshape([-1, NUM_DONG])
        node_daegu_infect_resahape = node_daegu_infect.reshape([-1, NUM_DONG])
        target_user_sty = []
        target_infect_sty = []
        for i, ymd_time in enumerate(ymd_time_lst):
            target_user_sty.append(node_daegu_user_reshape[i])
            target_infect_sty.append(node_daegu_infect_resahape[i])

        # mv_label                   
        mv_user = pd.DataFrame(np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3], edge_daegu_user[:,4]]).transpose(),
                               columns=['ymd_time', 'from_dong', 'to_dong', 'mv_num'])
        user_to_dong_grouped = mv_user.groupby(['ymd_time', 'to_dong']).sum()['mv_num']
        user_to_dong_grouped = user_to_dong_grouped.reset_index()

        raw_infect_mv_infect = pd.DataFrame(np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3], edge_daegu_infect[:,4]]).transpose(),
                                     columns=['ymd_time', 'from_dong', 'to_dong', 'mv_num'])
        infect_mv_infect = raw_infect_mv_infect.loc[1:]
        infect_to_dong_grouped = infect_mv_infect.groupby(['ymd_time', 'to_dong']).sum()['mv_num']
        infect_to_dong_grouped = infect_to_dong_grouped.reset_index()

        target_user_mv = []
        target_infect_mv = []
        for i, ymd_time in tqdm(enumerate(ymd_time_lst)):
            target_user_mv_dong, target_infect_mv_dong = torch.zeros([NUM_DONG]), torch.zeros([NUM_DONG])
            for j, dong in enumerate(dong_lst):
                if not user_to_dong_grouped[(user_to_dong_grouped['ymd_time']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)].empty:
                    target_user_mv_dong[j] = user_to_dong_grouped[(user_to_dong_grouped['ymd_time']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)]['mv_num'].item()
                if not infect_to_dong_grouped[(infect_to_dong_grouped['ymd_time']==ymd_time) & (infect_to_dong_grouped['to_dong']==dong)].empty:
                    target_infect_mv_dong[j] = infect_to_dong_grouped[(infect_to_dong_grouped['ymd_time']==ymd_time) & (infect_to_dong_grouped['to_dong']==dong)]['mv_num'].item()
            target_user_mv.append(target_user_mv_dong)
            target_infect_mv.append(target_infect_mv_dong)

        train_target, test_target = [], []
        for i in range(48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS), 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)):
            train_target_user = target_user_sty[i] + target_user_mv[i]
            train_target_infect = target_infect_sty[i] + target_infect_mv[i]
            train_target.append([train_target_user, train_target_infect])

        for i in range(48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS), 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS + TEST_DAYS)):
            test_target_user = target_user_sty[i] + target_user_mv[i]
            test_target_infect = target_infect_sty[i] + target_infect_mv[i]
            test_target.append([test_target_user, test_target_infect])                     

        # train datalaoder
        train_dataset = list(zip(train_inputs, train_target)) 
        test_dataset = list(zip(test_inputs, test_target))                                        

        with open(os.path.join(args.data_path + 'train_' + data_filename), 'wb') as f:
            torch.save(train_dataset, f)
        with open(os.path.join(args.data_path + 'test_' + data_filename), 'wb') as f:
            torch.save(test_dataset, f)
    
    else:
        train_dataset = torch.load(args.data_path + 'train_' + data_filename)
        test_dataset = torch.load(args.data_path + 'test_' + data_filename)

    train_loader = DataLoader(train_dataset,  shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)
    print('=== data preprocess done ===')
    
    return train_dataset, train_loader, test_dataset, test_loader, min_max_values

if __name__ == '__main__':
    import easydict
    args = easydict.EasyDict()
    args['device'] = 'cuda:0'
    args['train_days'] = 21
    args['delay_days'] = 0
    args['test_days'] = 7
    args['num_epochs'] = 250
    args['learning_rate'] = 1e-3
    args['num_dong'] = 139
    args['lstm_num_layers'] = 1
    args['lstm_input_size'] = 50
    args['lstm_hidden_size'] = 100
    args['lstm_sequence_length'] = 1
    args['graph_conv_feature_dim_infect'] = 20
    args['graph_conv_feature_dim_user'] = 20
    args['deep_graph'] = True
    args['concat_fc'] = True
    args['temp_all_concat'] = False
    args['all_concat'] = False
    args['loss'] = 'mse'
    args['task'] = 'real_feb'
    args['data_path'] = '/home/hyungjun/jupyter/KT_covid19/data/'
    args['data_normalize'] = True
    args['batch_size']=16
    
    train_dataset, train_loader, test_dataset, test_loader, min_max_values = data_loader_covid(args)
    
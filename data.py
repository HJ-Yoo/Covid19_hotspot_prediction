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


def data_loader(args):
    data_filename = os.path.join('norm()_d{}-{}-{}_s{}'.format(args.data_normalize, args.train_days, args.delay_days, args.test_days, args.lstm_sequence_length))
    
    if not os.path.exists(os.path.join(args.data_path + 'train_'+ data_filename)):
        # Dataset import 
        edge_daegu_infect = np.load('./data/raw_data/edge_daegu_infect.npy', allow_pickle=True)
        edge_daegu_user_1st = np.load('./data/raw_data/edge_daegu_user_1st.npy', allow_pickle=True)
        edge_daegu_user_2nd = np.load('./data/raw_data/edge_daegu_user_2nd.npy', allow_pickle=True)
        edge_daegu_user_3rd = np.load('./data/raw_data/edge_daegu_user_3rd.npy', allow_pickle=True)
        edge_daegu_user_4th = np.load('./data/raw_data/edge_daegu_user_4th.npy', allow_pickle=True)
        edge_daegu_user_5th = np.load('./data/raw_data/edge_daegu_user_5th.npy', allow_pickle=True)
        node_daegu_infect = torch.from_numpy(np.load('./data/raw_data/node_daegu_infect.npy', allow_pickle=True))
        node_daegu_user = torch.from_numpy(np.load('./data/raw_data/node_daegu_user.npy', allow_pickle=True))
        edge_daegu_user = np.concatenate((edge_daegu_user_1st,edge_daegu_user_2nd,edge_daegu_user_3rd,edge_daegu_user_4th,edge_daegu_user_5th))

        # hyperparameters
        LSTM_SEQUENCE_LENGTH = args.lstm_sequence_length
        TRAIN_DAYS = args.train_days
        DELAY_DAYS = args.delay_days
        TEST_DAYS = args.test_days
        DEVICE = args.device
        NUM_EPOCHS = args.num_epochs
        LEARNING_RATE = args.learning_rate

        # normalize (node, edge)
        if args.data_normalize:
            node_user = (node_daegu_user.reshape([-1, 142])-torch.min(node_daegu_user))/(torch.max(node_daegu_user)-torch.min(node_daegu_user))
            node_infect = (node_daegu_infect.reshape([-1, 142])-torch.min(node_daegu_infect))/(torch.max(node_daegu_infect)-torch.min(node_daegu_infect))

            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3],
                                  (edge_daegu_user[:,4]-np.min(edge_daegu_user[:,4]))/(np.max(edge_daegu_user[:,4])-np.min(edge_daegu_user[:,4]))]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3],
                                    (edge_daegu_infect[:,4]-np.min(edge_daegu_infect[:,4]))/(np.max(edge_daegu_infect[:,4])-np.min(edge_daegu_infect[:,4]))]).transpose()
    
        else:
            node_user = node_daegu_user.reshape([-1, 142])
            node_infect = node_daegu_infect.reshape([-1, 142])
            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3], edge_daegu_user[:,4]]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3], edge_daegu_infect[:,4]]).transpose()
        
        print(node_user)
        print(node_user.shape)
        print(edge_user)
        print(edge_user.shape)
       
        
        # node ready
        print('=== node_ready ===')
        train_infect_node_list, train_user_node_list = [], []
        test_infect_node_list, test_user_node_list = [], []
        for i in range(24*TRAIN_DAYS):
            train_infect_node_list.append(node_infect[i:i+(24*LSTM_SEQUENCE_LENGTH)])
            train_user_node_list.append(node_user[i:i+(24*LSTM_SEQUENCE_LENGTH)])

        for i in range(24*TRAIN_DAYS, 24*(TRAIN_DAYS + TEST_DAYS)):
            test_infect_node_list.append(node_infect[i:i+(24*LSTM_SEQUENCE_LENGTH)])
            test_user_node_list.append(node_user[i:i+(24*LSTM_SEQUENCE_LENGTH)])

        # edge ready
        print('=== edge_ready ===')
        ymd_time_lst = np.sort(np.unique(edge_user[:,0]))
        dong_lst = np.sort(np.unique(edge_daegu_user[:,2]))

        user_edge_inputs, infect_edge_inputs = [], []
        for ymd_time in tqdm(ymd_time_lst):
            g = nx.DiGraph()
            g.add_edges_from(edge_user[(edge_user[:,0]==ymd_time)][:,1:3])
            user_edge_list = torch_geometric.utils.from_networkx(g)
            user_edge_weight = edge_user[(edge_user[:,0]==ymd_time)][:,3]
            user_edge_inputs.append([user_edge_list, user_edge_weight])

        for ymd_time in tqdm(ymd_time_lst):
            g = nx.DiGraph()
            g.add_edges_from(edge_user[(edge_user[:,0]==ymd_time)][:,1:3])
            infect_edge_list = torch_geometric.utils.from_networkx(g)
            infect_edge_weight = edge_user[(edge_user[:,0]==ymd_time)][:,3]
            infect_edge_inputs.append([infect_edge_list, infect_edge_weight])

        train_user_edge_inputs, train_infect_edge_inputs = [], []
        test_user_edge_inputs, test_infect_edge_inputs = [], []
        for i in range(24 * TRAIN_DAYS):
            train_user_edge_inputs.append(user_edge_inputs[i:i+(24*LSTM_SEQUENCE_LENGTH)])
            train_infect_edge_inputs.append(infect_edge_inputs[i:i+(24*LSTM_SEQUENCE_LENGTH)])

        for i in range(24 * TRAIN_DAYS, 24 * (TRAIN_DAYS + TEST_DAYS)):
            test_user_edge_inputs.append(user_edge_inputs[i:i+(24*LSTM_SEQUENCE_LENGTH)])
            test_infect_edge_inputs.append(infect_edge_inputs[i:i+(24*LSTM_SEQUENCE_LENGTH)])

        ###########################    
        ##### temporal signal #####

        weekday_dict = {'sat':0, 'sun':1, 'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6}
        weekday_list = ['sat', 'sun', 'mon', 'tue','wed','thu','fri']
        time_list = ['00','01','02','03','04','05','06','07','08','09',
                    '10','11','12','13','14','15','16','17','18','19',
                    '20','21','22','23']

        temp_sig = torch.zeros([29*24, 7 + 24 + 1 + 1]) 

        for i in range(4): # 4 weeks
            for weekday in weekday_list:
                for time in range(24):
                    temp_sig[168*i + 24*weekday_dict[weekday] + time][weekday_dict[weekday]] = 1
                    temp_sig[168*i + 24*weekday_dict[weekday] + time][7 + time] = 1
                    if weekday_dict[weekday] == 6: # 금요일
                        temp_sig[168*i + 24*weekday_dict[weekday] + time][7+24+1] = 1 # before holiday
                    elif weekday_dict[weekday] == 0: # 토요일
                        temp_sig[168*i + 24*weekday_dict[weekday] + time][7+24+1] = 1 # before holiday
                        temp_sig[168*i + 24*weekday_dict[weekday] + time][7+24] = 1 # holiday
                    elif weekday_dict[weekday] == 1: # 일요일
                        temp_sig[168*i + 24*weekday_dict[weekday] + time][7+24] = 1 # holiday

        # the last day(29th)
        for time in range(24):
            temp_sig[672 + time][weekday_dict['sat']] = 1
            temp_sig[672 + time][7 + time] = 1
            temp_sig[672 + time][7+24+1] = 1 # before holiday
            temp_sig[672 + time][7+24] = 1 # holiday

        train_temp_sig_inputs, test_temp_sig_inputs = [], []
        for i in range(24 * TRAIN_DAYS):
            train_temp_sig_inputs.append(temp_sig[i:i+24*LSTM_SEQUENCE_LENGTH])

        for i in range(24 * TRAIN_DAYS, 24 * (TRAIN_DAYS + TEST_DAYS)):
            test_temp_sig_inputs.append(temp_sig[i:i+24*LSTM_SEQUENCE_LENGTH])


        # input concatenate
        print("=== concat ===")
        train_inputs = []
        for i in tqdm(range(24*TRAIN_DAYS)):
            train_g_user_inputs, train_g_infect_inputs = [], []
            for j in range(24*LSTM_SEQUENCE_LENGTH):
                train_g_user_inputs.append([train_user_node_list[i][j], train_user_edge_inputs[i][j][0].edge_index, 
                                            torch.tensor(train_user_edge_inputs[i][j][1].astype(float))])
                train_g_infect_inputs.append([train_infect_node_list[i][j], train_infect_edge_inputs[i][j][0].edge_index, 
                                              torch.tensor(train_infect_edge_inputs[i][j][1].astype(float))])
            train_inputs.append([train_g_user_inputs, train_g_infect_inputs, train_temp_sig_inputs[i]])

        test_inputs = []
        for i in tqdm(range(24*TEST_DAYS)):
            test_g_user_inputs, test_g_infect_inputs = [], []
            for j in range(24*LSTM_SEQUENCE_LENGTH):
                test_g_user_inputs.append([test_user_node_list[i][j], test_user_edge_inputs[i][j][0].edge_index, 
                                           torch.tensor(test_user_edge_inputs[i][j][1].astype(float))])
                test_g_infect_inputs.append([test_infect_node_list[i][j], test_infect_edge_inputs[i][j][0].edge_index, 
                                             torch.tensor(test_infect_edge_inputs[i][j][1].astype(float))])
            test_inputs.append([test_g_user_inputs, test_g_infect_inputs, test_temp_sig_inputs[i]])

        # sty infect (label)
        target_sty = node_daegu_infect.reshape([-1,142])
        train_target_sty = target_sty[24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
        test_target_sty = target_sty[24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]
        # mv infect (label)
        raw_mv_infect = pd.DataFrame(edge_infect, columns=['ymd_time', 'from_dong', 'to_dong', 'mv_num'])
        mv_infect = raw_mv_infect.loc[1:]
        to_dong_grouped = mv_infect.groupby(['ymd_time', 'to_dong']).sum()['mv_num']
        to_dong_grouped = to_dong_grouped.reset_index()
        
        target_mv = torch.zeros([696, 142])
        for i, ymd_time in tqdm(enumerate(ymd_time_lst)):
            for j, dong in enumerate(dong_lst):
                if not to_dong_grouped[(to_dong_grouped['ymd_time']==ymd_time) & (to_dong_grouped['to_dong']==dong)].empty:
                    target_mv[i][j] = to_dong_grouped[(to_dong_grouped['ymd_time']==ymd_time) & (to_dong_grouped['to_dong']==dong)]['mv_num'].item()
        train_target_mv = target_mv[24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
        test_target_mv = target_mv[24*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]
                    
        train_target = train_target_mv + train_target_sty
        test_target = test_target_mv + test_target_sty
        
        # train datalaoder
        train_dataset = list(zip(train_inputs, train_target)) 
        test_dataset = list(zip(test_inputs, test_target))
        
        # with open(os.path.join(args.data_path + 'temp_train_target_mv'), 'wb') as f:
        #     torch.save(train_target_mv, f)
            
        # with open(os.path.join(args.data_path + 'temp_train_target_sty.csv'), 'wb') as f:
        #     torch.save(train_target_sty, f)                                                   
        
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
    
    return train_dataset, train_loader, test_dataset, test_loader
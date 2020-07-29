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


def data_loader_half(args):
    data_filename = os.path.join('{}_norm{}_d{}-{}-{}_s{}'.format(args.task, args.data_normalize, args.train_days, args.delay_days, args.test_days, args.lstm_sequence_length))
    
    if not os.path.exists(os.path.join(args.data_path + 'train_'+ data_filename)):
        print('== data import ==')
        # Dataset import 
        edge_daegu_infect = np.load('./data/raw_data/half/edge_daegu_infect_half_feb.npy', allow_pickle=True)
        edge_daegu_fake_infect_full = np.load('./data/raw_data/half/edge_daegu_fake_infect_half_full.npy', allow_pickle=True)
        edge_daegu_user = np.load('./data/raw_data/half/edge_daegu_user_half_feb.npy', allow_pickle=True) # 2월
        edge_daegu_user_mar = np.load('./data/raw_data/half/edge_daegu_user_half_mar.npy', allow_pickle=True) 
        edge_daegu_user_full = np.concatenate((edge_daegu_user, edge_daegu_user_mar))
        node_daegu_fake_infect_full = torch.from_numpy(np.load('./data/raw_data/half/node_daegu_fake_infect_half_mar.npy', allow_pickle=True))
        node_daegu_infect = torch.from_numpy(np.load('./data/raw_data/half/node_daegu_fake_infect_half_feb.npy', allow_pickle=True))
        node_daegu_user = torch.from_numpy(np.load('./data/raw_data/half/node_daegu_user_half_feb.npy', allow_pickle=True)) # 2월
        node_daegu_user_mar = torch.from_numpy(np.load('./data/raw_data/half/node_daegu_user_half_mar.npy', allow_pickle=True))
        node_daegu_user_full = torch.cat((node_daegu_user, node_daegu_user_mar), dim=0)

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
            node_user_full = (node_daegu_user_full.reshape([-1, NUM_DONG])-torch.min(node_daegu_user_full))/ \
                             (torch.max(node_daegu_user_full)-torch.min(node_daegu_user_full))
            node_fake_infect_full = (node_daegu_fake_infect_full.reshape([-1, NUM_DONG])-torch.min(node_daegu_fake_infect_full))/ \
                                    (torch.max(node_daegu_fake_infect_full)-torch.min(node_daegu_fake_infect_full))

            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3],
                                  (edge_daegu_user[:,4]-np.min(edge_daegu_user[:,4]))/ \
                                  (np.max(edge_daegu_user[:,4])-np.min(edge_daegu_user[:,4]))]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3],
                                    (edge_daegu_infect[:,4]-np.min(edge_daegu_infect[:,4]))/ \
                                    (np.max(edge_daegu_infect[:,4])-np.min(edge_daegu_infect[:,4]))]).transpose()
            edge_user_full = np.array([edge_daegu_user_full[:,0]+edge_daegu_user_full[:,1], edge_daegu_user_full[:,2], edge_daegu_user_full[:,3],
                                      (edge_daegu_user_full[:,4]-np.min(edge_daegu_user_full[:,4]))/ \
                                      (np.max(edge_daegu_user_full[:,4])-np.min(edge_daegu_user_full[:,4]))]).transpose()
            edge_fake_infect_full = np.array([edge_daegu_fake_infect_full[:,0]+edge_daegu_fake_infect_full[:,1], edge_daegu_fake_infect_full[:,2],
                                              edge_daegu_fake_infect_full[:,3],(edge_daegu_fake_infect_full[:,4] - np.min(edge_daegu_fake_infect_full[:,4]))/ \
                                              (np.max(edge_daegu_fake_infect_full[:,4])-np.min(edge_daegu_fake_infect_full[:,4]))]).transpose()
            
            # order : user, infect , user_full, infect_full
            self.min_max_values = [torch.min(node_daegu_user).item(), torch.max(node_daegu_user).item(), 
                                  np.min(edge_daegu_user[:,4]), np.max(edge_daegu_user[:,4]), 
                                  torch.min(node_daegu_infect).item(), torch.max(node_daegu_infect).item(),
                                  np.min(edge_daegu_user[:,4]), np.max(edge_daegu_user[:,4]), 
                                  torch.min(node_daegu_user_full).item(), torch.max(node_daegu_user_full).item(), 
                                  np.min(edge_daegu_user_full[:,4]), np.max(edge_daegu_user_full[:,4]), 
                                  torch.min(node_daegu_fake_infect_full).item(), torch.max(node_daegu_fake_infect_full).item(),
                                  np.min(edge_daegu_fake_infect_full[:,4]), np.max(edge_daegu_fake_infect_full[:,4])]

        else:
            node_user = node_daegu_user.reshape([-1, NUM_DONG])
            node_infect = node_daegu_infect.reshape([-1, NUM_DONG])
            node_user_full = node_daegu_user_full.reshape([-1, NUM_DONG])
            node_fake_infect_full = node_daegu_fake_infect_full.reshape([-1, NUM_DONG])
            edge_user = np.array([edge_daegu_user[:,0]+edge_daegu_user[:,1], edge_daegu_user[:,2], edge_daegu_user[:,3], 
                                  edge_daegu_user[:,4]]).transpose()
            edge_infect = np.array([edge_daegu_infect[:,0]+edge_daegu_infect[:,1], edge_daegu_infect[:,2], edge_daegu_infect[:,3], 
                                    edge_daegu_infect[:,4]]).transpose()
            edge_user_full = np.array([edge_daegu_user_full[:,0]+edge_daegu_user_full[:,1], edge_daegu_user_full[:,2], edge_daegu_user_full[:,3], 
                                       edge_daegu_user_full[:,4]]).transpose()
            edge_fake_infect_full = np.array([edge_daegu_fake_infect_full[:,0]+edge_daegu_fake_infect_full[:,1], edge_daegu_fake_infect_full[:,2], 
                                              edge_daegu_fake_infect_full[:,3], edge_daegu_fake_infect_full[:,4]]).transpose()
        
        # node ready
        print('=== node_ready ===')
        train_infect_node_list, train_user_node_list = [], []
        train_fake_infect_full_node_list, train_user_full_node_list = [], []
        test_infect_node_list, test_user_node_list = [], []
        test_fake_infect_full_node_list, test_user_full_node_list = [], []

        for i in range(48*TRAIN_DAYS):
            train_infect_node_list.append(node_infect[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            train_user_node_list.append(node_user[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            train_fake_infect_full_node_list.append(node_fake_infect_full[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            train_user_full_node_list.append(node_user_full[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        for i in range(48*TRAIN_DAYS, 48*(TRAIN_DAYS + TEST_DAYS)):
            test_infect_node_list.append(node_infect[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            test_user_node_list.append(node_user[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            test_fake_infect_full_node_list.append(node_fake_infect_full[i:i+(48*LSTM_SEQUENCE_LENGTH)])
            test_user_full_node_list.append(node_user_full[i:i+(48*LSTM_SEQUENCE_LENGTH)])

        # edge ready
        print('=== edge_ready ===')
        ymd_time_lst = np.sort(np.unique(edge_user[:,0]))
        ymd_time_full_lst = np.sort(np.unique(edge_user_full[:,0]))
        dong_lst = np.sort(np.unique(edge_daegu_user[:,2]))

        user_edge_inputs, infect_edge_inputs = [], []
        user_full_edge_inputs, fake_infect_full_edge_inputs = [], []

        # real feb
        if args.task == 'real_feb':
            print('graph making, task : real_feb')
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

        # fake full
        elif args.task == 'fake_full':
            print('graph making, task : fake_full')
            for ymd_time in tqdm(ymd_time_full_lst):
                g = nx.DiGraph()
                g.add_edges_from(edge_user_full[(edge_user_full[:,0]==ymd_time)][:,1:3])
                user_full_edge_list = torch_geometric.utils.from_networkx(g)
                user_full_edge_weight = edge_user_full[(edge_user_full[:,0]==ymd_time)][:,3]
                user_full_edge_inputs.append([user_full_edge_list, user_full_edge_weight])

            for ymd_time in tqdm(ymd_time_full_lst):
                g = nx.DiGraph()
                g.add_edges_from(edge_fake_infect_full[(edge_fake_infect_full[:,0]==ymd_time)][:,1:3])
                fake_infect_full_edge_list = torch_geometric.utils.from_networkx(g)
                fake_infect_full_edge_weight = edge_fake_infect_full[(edge_fake_infect_full[:,0]==ymd_time)][:,3]
                fake_infect_full_edge_inputs.append([fake_infect_full_edge_list, fake_infect_full_edge_weight])

            train_user_full_edge_inputs, train_fake_infect_full_edge_inputs = [], []
            test_user_full_edge_inputs, test_fake_infect_full_edge_inputs = [], []

            for i in range(48 * TRAIN_DAYS):
                train_user_full_edge_inputs.append(user_full_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])
                train_fake_infect_full_edge_inputs.append(fake_infect_full_edge_inputs[i:i*(48*LSTM_SEQUENCE_LENGTH)])

            for i in range(48*TRAIN_DAYS, 48*(TRAIN_DAYS + TEST_DAYS)):
                test_user_full_edge_inputs.append(user_full_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])
                test_fake_infect_full_edge_inputs.append(fake_infect_full_edge_inputs[i:i+(48*LSTM_SEQUENCE_LENGTH)])
        
        # temporal signal
        print('== temporal_signal ready ==')
        weekday_list = ['sat', 'sun', 'mon', 'tue','wed','thu','fri']
        
        temp_sig = torch.zeros([29*48, 7 + 48 + 1 + 1]) 
        for i in range(4): # 4 weeks
            for j, weekday in enumerate(weekday_list):
                for time in range(48):
                    temp_sig[(48*7)*i + 48*j + time][i] = 1
                    temp_sig[(48*7)*i + 48*j + time][7 + time] = 1
                    if i == 6: # 금요일
                        temp_sig[(48*7)*i + 48*j + time][7+48+1] = 1 # before holiday
                    elif i == 0: # 토요일
                        temp_sig[(48*7) + 48*j + time][7+48+1] = 1 # before holiday
                        temp_sig[(48*7) + 48*j + time][7+48] = 1 # holiday
                    elif i == 1: # 일요일
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

        # full
        temp_sig_full = torch.zeros([(29+31)*48, 7 + 48 + 1 + 1])

        for i in range(8): # 8 weeks
            for j, weekday in enumerate(weekday_list):
                for time in range(48):
                    temp_sig_full[(48*7)*i + 48*j + time][i] = 1
                    temp_sig_full[(48*7)*i + 48*j + time][7 + time] = 1
                    if i == 6: # 금요일
                        temp_sig_full[(48*7)*i + 48*j + time][7+48+1] = 1 # before holiday
                    elif i == 0: # 토요일
                        temp_sig_full[(48*7)*i + 48*j + time][7+48+1] = 1 # before holiday
                        temp_sig_full[(48*7)*i + 48*j + time][7+48] = 1 # holiday
                    elif i == 1: # 일요일
                        temp_sig_full[(48*7)*i + 48*j + time][7+48] = 1 # holiday

        # last 4 days
        for j in range(4):
            for time in range(48):
                temp_sig_full[56*48 + time][j] = 1
                temp_sig_full[56*48 + time][7 + time] = 1
                if i == 6: # 금요일
                    temp_sig_full[(48*7)*7 + 48*j + time][7+48+1] = 1 # before holiday
                elif i == 0: # 토요일
                    temp_sig_full[(48*7)*7 + 48*j + time][7+48+1] = 1 # before holiday
                    temp_sig_full[(48*7)*7 + 48*j + time][7+48] = 1 # holiday
                elif i == 1: # 일요일
                    temp_sig_full[(48*7)*7 + 48*j + time][7+48] = 1 # holiday

        train_temp_sig_full_inputs, test_temp_sig_full_inputs = [], []
        for i in range(48 * TRAIN_DAYS):
            train_temp_sig_full_inputs.append(temp_sig_full[i:i+48*LSTM_SEQUENCE_LENGTH])

        for i in range(48 * TRAIN_DAYS, 48 * (TRAIN_DAYS + TEST_DAYS)):
            test_temp_sig_full_inputs.append(temp_sig_full[i:i+48*LSTM_SEQUENCE_LENGTH])

        # input concatenate
        print("=== concat ===")
        if args.task == 'real_feb':
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
        
        elif args.task == 'fake_full':
            train_full_inputs = []
            for i in tqdm(range(48*TRAIN_DAYS)):
                train_g_user_full_inputs, train_g_fake_infect_full_inputs = [], []
                for j in range(48*LSTM_SEQUENCE_LENGTH):
                    train_g_user_full_inputs.append([train_user_full_node_list[i][j], train_user_full_edge_inputs[i][j][0].edge_index, 
                                                        torch.tensor(train_user_full_edge_inputs[i][j][1].astype(float))])
                    train_g_fake_infect_full_inputs.append([train_fake_infect_full_node_list[i][j], train_fake_infect_full_edge_inputs[i][j][0].edge_index, 
                                                          torch.tensor(train_fake_infect_full_edge_inputs[i][j][1].astype(float))])
                train_full_inputs.append([train_g_user_full_inputs, train_g_fake_infect_full_inputs, train_temp_sig_full_inputs[i]])

            test_full_inputs = []
            for i in tqdm(range(48*TEST_DAYS)):
                test_g_user_full_inputs, test_g_fake_infect_full_inputs = [], []
                for j in range(48*LSTM_SEQUENCE_LENGTH):
                    test_g_user_full_inputs.append([test_user_full_node_list[i][j], test_user_full_edge_inputs[i][j][0].edge_index, 
                                                       torch.tensor(test_user_full_edge_inputs[i][j][1].astype(float))])
                    test_g_fake_infect_full_inputs.append([test_fake_infect_full_node_list[i][j], test_fake_infect_full_edge_inputs[i][j][0].edge_index, 
                                                         torch.tensor(test_fake_infect_full_edge_inputs[i][j][1].astype(float))])
                test_full_inputs.append([test_g_user_full_inputs, test_g_fake_infect_full_inputs, test_temp_sig_full_inputs[i]])

        ###################
        ###### label ######
        print('== label ready ==')
        if args.task == 'real_feb':
            # user, infect
            # sty label
            target_user_sty = node_daegu_user.reshape([-1, NUM_DONG])
            train_target_user_sty = target_user_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_user_sty = target_user_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

            target_infect_sty = node_daegu_infect.reshape([-1, NUM_DONG])
            train_target_infect_sty = target_infect_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_infect_sty = target_infect_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

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

            target_user_mv = torch.zeros([48*(TRAIN_DAYS+DELAY_DAYS+TEST_DAYS), NUM_DONG])
            target_infect_mv = torch.zeros([48*(TRAIN_DAYS+DELAY_DAYS+TEST_DAYS), NUM_DONG])
            for i, ymd_time in tqdm(enumerate(ymd_time_lst)):
                for j, dong in enumerate(dong_lst):
                    if not user_to_dong_grouped[(user_to_dong_grouped['ymd_time']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)].empty:
                        target_user_mv[i][j] = user_to_dong_grouped[(user_to_dong_grouped['ymd_time']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)]['mv_num'].item()
                    if not infect_to_dong_grouped[(infect_to_dong_grouped['ymd_time']==ymd_time) & (infect_to_dong_grouped['to_dong']==dong)].empty:
                        target_infect_mv[i][j] = infect_to_dong_grouped[(infect_to_dong_grouped['ymd_time']==ymd_time) & (infect_to_dong_grouped['to_dong']==dong)]['mv_num'].item()

            train_target_user_mv = target_user_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_user_mv = target_user_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]
            train_target_infect_mv = target_infect_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_infect_mv = target_infect_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

            train_target = [train_target_user_sty + train_target_user_mv, train_target_infect_sty + train_target_infect_mv]
            test_target = [test_target_user_sty + test_target_user_mv, test_target_infect_sty + test_target_infect_mv]                           

            # train datalaoder
            train_dataset = list(zip(train_inputs, train_target)) 
            test_dataset = list(zip(test_inputs, test_target))
                           
        elif args.task == 'fake_full':
            # user_full, fake_infect_full
            # sty label
            target_user_full_sty = node_daegu_user_full.reshape([-1, NUM_DONG])
            train_target_user_full_sty = target_user_full_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_user_full_sty = target_user_full_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

            target_fake_infect_full_sty = node_daegu_fake_infect_full.reshape([-1, NUM_DONG])
            train_target_fake_infect_full_sty = target_fake_infect_full_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_fake_infect_full_sty = target_fake_infect_full_sty[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

            # mv label
            mv_user_full = pd.DataFrame(np.array([edge_daegu_user_full[:,0]+edge_daegu_user_full[:,1], edge_daegu_user_full[:,2], edge_daegu_user_full[:,3], edge_daegu_user_full[:,4]]).transpose(),
                                   columns=['ymd_time', 'from_dong', 'to_dong', 'mv_num'])
            user_full_to_dong_grouped = mv_user_full.groupby(['ymd_time', 'to_dong']).sum()['mv_num']
            user_full_to_dong_grouped = user_full_to_dong_grouped.reset_index()

            fake_infect_full_mv_infect = pd.DataFrame(np.array([edge_daegu_fake_infect_full[:,0]+edge_daegu_fake_infect_full[:,1], edge_daegu_fake_infect_full[:,2], edge_daegu_fake_infect_full[:,3], edge_daegu_fake_infect_full[:,4]]).transpose(),
                                         columns=['ymd_time', 'from_dong', 'to_dong', 'mv_num'])
            fake_infect_full_to_dong_grouped = fake_infect_full_mv_infect.groupby(['ymd_time', 'to_dong']).sum()['mv_num']
            fake_infect_full_to_dong_grouped = fake_infect_full_to_dong_grouped.reset_index()

            target_user_full_mv = torch.zeros([48*(TRAIN_DAYS+DELAY_DAYS+TEST_DAYS), NUM_DONG])
            target_fake_infect_full_mv = torch.zeros([48*(TRAIN_DAYS+DELAY_DAYS+TEST_DAYS), NUM_DONG])
            for i, ymd_time in tqdm(enumerate(ymd_time_full_lst)):
                for j, dong in enumerate(dong_lst):
                    if not user_full_to_dong_grouped[(user_full_to_dong_grouped['ymd_time']==ymd_time) & (user_full_to_dong_grouped['to_dong']==dong)].empty:
                        target_user_full_mv[i][j] = user_full_to_dong_grouped[(user_full_to_dong_grouped['ymd_time']==ymd_time) & (user_full_to_dong_grouped['to_dong']==dong)]['mv_num'].item()
                    if not fake_infect_full_to_dong_grouped[(fake_infect_full_to_dong_grouped['ymd_time']==ymd_time) & (fake_infect_full_to_dong_grouped['to_dong']==dong)].empty:
                        target_fake_infect_full_mv[i][j] = fake_infect_full_to_dong_grouped[(fake_infect_full_to_dong_grouped['ymd_time']==ymd_time) & (fake_infect_full_to_dong_grouped['to_dong']==dong)]['mv_num'].item()

            train_target_user_full_mv = target_user_full_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_user_full_mv = target_user_full_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]
            train_target_fake_infect_full_mv = target_fake_infect_full_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS) : 48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)]
            test_target_fake_infect_full_mv = target_fake_infect_full_mv[48*(LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS):]

            train_full_target = [train_target_user_full_sty + train_target_user_full_mv, train_target_fake_infect_full_sty + train_target_fake_infect_full_mv]
            test_full_target = [test_target_user_full_sty + test_target_user_full_mv, test_target_fake_infect_full_sty + test_target_fake_infect_full_mv]

            train_dataset = list(zip(train_full_inputs, train_full_target))
            test_dataset = list(zip(test_full_inputs, test_full_target))                                               

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
    
    return train_dataset, train_loader, test_dataset, test_loader, self.min_max_values
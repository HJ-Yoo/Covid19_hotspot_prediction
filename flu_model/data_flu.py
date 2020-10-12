import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import networkx as nx
import torch_geometric
import os

def test_check(data):
    if data[4] == 'Not Done':
        if data[6] =='Not Done' or data[6] == 'No Test':
            return None
        return data[6].split('-')
    else:
        return data[4].split('-')

def data_loader_flu(args):
    data_filename = os.path.join('flu_norm{}_d{}-{}-{}_s{}'.format(args.data_normalize, args.train_days, args.delay_days, args.test_days, args.lstm_sequence_length))
    
    print('== data import ==')
    # Dataset import -------------------------------------------------------------------------------------
    # 모든 전처리 완료된 데이터는 './data/raw_data' 아래에
    node_seoul_user_17_np = np.load(args.data_path + 'raw_data/flu/seoul_sgg_sty_2017_2018.npy', allow_pickle=True)
    node_seoul_user_18_np = np.load(args.data_path + 'raw_data/flu/seoul_sgg_sty_2018_2019.npy', allow_pickle=True) 
    node_seoul_user_19_np = np.load(args.data_path + 'raw_data/flu/seoul_sgg_sty_2019_2020.npy', allow_pickle=True)
    node_seoul_user_17 = torch.tensor(node_seoul_user_17_np[:,2].reshape([-1, 25]).astype(float))
    node_seoul_user_18 = torch.tensor(node_seoul_user_18_np[:,2].reshape([-1, 25]).astype(float))
    node_seoul_user_19 = torch.tensor(node_seoul_user_19_np[:,2].reshape([-1, 25]).astype(float))
    node_seoul_user = torch.cat((node_seoul_user_17, node_seoul_user_18, node_seoul_user_19), dim=0)
    
    edge_seoul_user_17 = np.load(args.data_path + 'raw_data/flu/seoul_sgg_mv_2017_2018.npy', allow_pickle=True)
    edge_seoul_user_18 = np.load(args.data_path + 'raw_data/flu/seoul_sgg_mv_2018_2019.npy', allow_pickle=True)
    edge_seoul_user_19 = np.load(args.data_path + 'raw_data/flu/seoul_sgg_mv_2019_2020.npy', allow_pickle=True) 
    edge_seoul_user = np.concatenate((edge_seoul_user_17, edge_seoul_user_18, edge_seoul_user_19))
    
    flu = np.load('./data/raw_data/flu/flu_np.npy', allow_pickle=True)
    
    # flu data reconstruction
    flu_seoul = flu[flu[:,2]=='seoul']

    flu_seoul_lst = flu_seoul.tolist()
    cols = np.unique(flu_seoul[:,3])
    np.where(flu_seoul[:,3]=='yongsan ')
    flu_seoul_lst[547][3] = 'yongsan'
    np.where(flu_seoul[:,3]=='dongjak ')
    flu_seoul_lst[447][3] = 'dongjak'
    np.where(flu_seoul[:,3]=='jongno ')
    flu_seoul_lst[545][3] = 'jongno'
    np.where(flu_seoul[:,3]=='gwangmyeong')
    del flu_seoul_lst[672]

    flu_seoul = np.array(flu_seoul_lst)
    cols = np.unique(flu_seoul[:,3])
    cols = np.append(cols, 'junggu')
    cols = np.append(cols, 'sungdong')
    cols = sorted(cols)

    idx_17_18 = [datetime.datetime(year=2017, month=11, day=1)+datetime.timedelta(days=i) for i in range(151)]
    idx_18_19 = [datetime.datetime(year=2018, month=11, day=1)+datetime.timedelta(days=i) for i in range(151)]
    idx_19_20 = [datetime.datetime(year=2019, month=11, day=1)+datetime.timedelta(days=i) for i in range(152)]
    idx = idx_17_18 + idx_18_19 + idx_19_20

    flu_df = pd.DataFrame(data = np.zeros([len(idx), len(cols)]), index=idx, columns=cols)
    for i in range(len(flu_seoul)):
        date = test_check(flu_seoul[i,:])
        if date ==None:
            continue
        year, mon, day = int(date[0]), int(date[1]), int(date[2])
        date = datetime.datetime(year=year, month=mon, day=day)
        sgg = flu_seoul[i,:][3]
        if date not in idx:
            continue
        for k in range(5):
            curr_date = date+datetime.timedelta(days=k)
            if curr_date not in idx:
                continue
            flu_df.loc[date+datetime.timedelta(days=k),sgg] += 1

    np.save('./data/raw_data/flu/flu_seoul.npy',flu_df.to_numpy())
    
    seoul_flu = torch.from_numpy(np.load(args.data_path + '/raw_data/flu/flu_seoul.npy', allow_pickle=True))
    
    # order : user, infect , user_full, infect_full
    min_max_values = [torch.min(node_seoul_user).item(), torch.max(node_seoul_user).item(), 
                      np.min(edge_seoul_user[:,3]), np.max(edge_seoul_user[:,3]), 
                      torch.min(seoul_flu).item(), torch.max(seoul_flu).item()]
    
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
            node_user = (node_seoul_user.reshape([-1, NUM_DONG])-torch.min(node_seoul_user))/(torch.max(node_seoul_user)-torch.min(node_seoul_user))
            edge_user = np.array([edge_seoul_user[:,0], edge_seoul_user[:,1], edge_seoul_user[:,2],
                                  (edge_seoul_user[:,3]-np.min(edge_seoul_user[:,3]))/ \
                                  (np.max(edge_seoul_user[:,3])-np.min(edge_seoul_user[:,3]))]).transpose()
        else:
            node_user = node_daegu_user.reshape([-1, NUM_DONG])
            
        # node ready
        print('=== node_ready ===')
        train_user_node_list, test_user_node_list= [], []
        
        for i in range(TRAIN_DAYS):
            train_user_node_list.append(node_user[i:i+(LSTM_SEQUENCE_LENGTH)])
            
        for i in range(TRAIN_DAYS, (TRAIN_DAYS + TEST_DAYS)):
            test_user_node_list.append(node_user[i:i+(LSTM_SEQUENCE_LENGTH)])
            
        # edge ready
        print('=== edge_ready ===')
        ymd_time_lst = np.sort(np.unique(edge_user[:,0]))
        dong_lst = np.sort(np.unique(edge_user[:,2]))

        user_edge_inputs, infect_edge_inputs = [], []
        user_full_edge_inputs, fake_infect_full_edge_inputs = [], []
        
        # edge_list, edge_weight generation with network X
        for ymd_time in ymd_time_lst:
            g = nx.DiGraph()
            g.add_edges_from(edge_user[(edge_user[:,0]==ymd_time)][:,1:3])
            user_edge_list = torch_geometric.utils.from_networkx(g)
            user_edge_weight = edge_user[(edge_user[:,0]==ymd_time)][:,3]
            user_edge_inputs.append([user_edge_list, user_edge_weight])

        train_user_edge_inputs, test_user_edge_inputs = [], []

        for i in range(TRAIN_DAYS):
            train_user_edge_inputs.append(user_edge_inputs[i:i+(LSTM_SEQUENCE_LENGTH)])

        for i in range(TRAIN_DAYS, (TRAIN_DAYS + TEST_DAYS)):
            test_user_edge_inputs.append(user_edge_inputs[i:i+(LSTM_SEQUENCE_LENGTH)])
                
        # temporal signal & weather -------------------------------------------------------------------------------------
        print('== temporal_signal ready ==')
        weather_17 = pd.read_csv(args.data_path + 'raw_data/flu/2017_2018_weather.csv', sep=',', encoding='iso-8859-1')
        weather_18 = pd.read_csv(args.data_path + 'raw_data/flu/2018_2019_weather.csv', sep=',', encoding='iso-8859-1')
        weather_19 = pd.read_csv(args.data_path + 'raw_data/flu/2019_2020_weather.csv', sep=',', encoding='iso-8859-1')

        weather_17.columns = ['지역', '지역명', '일시', '기온', '강수량','습도','적설량']
        del weather_17['지역']
        del weather_17['지역명']
        weather_17.loc[weather_17['강수량'].isna(), '강수량'] = 0
        weather_17.loc[weather_17['적설량'].isna(), '적설량'] = 0
        time_split = weather_17['일시'].str.split(" ")
        weather_17['etl_ymd'] = time_split.str.get(0)
        weather_17['time'] = time_split.str.get(1)
        weather_17 = weather_17.groupby('etl_ymd').mean().reset_index()

        weather_18.columns = ['지역', '지역명', '일시', '기온', '강수량','습도','적설량']
        del weather_18['지역']
        del weather_18['지역명']
        weather_18.loc[weather_18['강수량'].isna(), '강수량'] = 0
        weather_18.loc[weather_18['적설량'].isna(), '적설량'] = 0
        time_split = weather_18['일시'].str.split(" ")
        weather_18['etl_ymd'] = time_split.str.get(0)
        weather_18['time'] = time_split.str.get(1)
        weather_18 = weather_18.groupby('etl_ymd').mean().reset_index()

        weather_19.columns = ['지역', '지역명', '일시', '기온', '강수량','습도','적설량']
        del weather_19['지역']
        del weather_19['지역명']
        weather_19.loc[weather_19['강수량'].isna(), '강수량'] = 0
        weather_19.loc[weather_19['적설량'].isna(), '적설량'] = 0
        time_split = weather_19['일시'].str.split(" ")
        weather_19['etl_ymd'] = time_split.str.get(0)
        weather_19['time'] = time_split.str.get(1)
        weather_19 = weather_19.groupby('etl_ymd').mean().reset_index()

        weekday_list = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
        weekday_dict_17 = {'wed':3, 'thr':4, 'fri':5, 'sat':6, 'sun':0, 'mon':1, 'tue':2}
        weekday_dict_18 = {'thr':4, 'fri':5, 'sat':6, 'sun':0, 'mon':1, 'tue':2, 'wed':3}
        weekday_dict_19 = {'fri':5, 'sat':6, 'sun':0, 'mon':1, 'tue':2, 'wed':3, 'thr':4}
                
        temp_sig_17 = torch.zeros([30+31+31+28+31, 7 + 1 + 1 + 1 + 1 + 1 + 1]) 
        temp_sig_18 = torch.zeros([30+31+31+28+31, 7 + 1 + 1 + 1 + 1 + 1 + 1])
        temp_sig_19 = torch.zeros([30+31+31+29+31, 7 + 1 + 1 + 1 + 1 + 1 + 1])
        
        # 2017
        for i in range(21): # 21 weeks + a
            for j, (weekday, val) in enumerate(weekday_dict_17.items()):
                temp_sig_17[7*i + j][val] = 1 # weekday
                temp_sig_17[7*i + j][7+1+1] = weather_17.loc[7*i + j,'기온']
                temp_sig_17[7*i + j][7+1+1+1] = weather_17.loc[7*i + j,'강수량']
                temp_sig_17[7*i + j][7+1+1+1+1] = weather_17.loc[7*i + j,'습도']
                temp_sig_17[7*i + j][7+1+1+1+1+1] = weather_17.loc[7*i + j,'적설량']
                
                if weekday == 'fri': # 금요일
                    temp_sig_17[7*i + j][7+1] = 1 # before holiday
                elif weekday == 'sat': # 토요일
                    temp_sig_17[7*i + j][7+1] = 1 # before holiday
                    temp_sig_17[7*i + j][7] = 1 # holiday
                elif weekday == 'sun': # 일요일
                    temp_sig_17[7*i + j][7] = 1 # holiday
        # 2018    
        for i in range(21): # 21 weeks + a
            for j, (weekday, val) in enumerate(weekday_dict_17.items()):
                temp_sig_18[7*i + j][val] = 1 # weekday
                temp_sig_18[7*i + j][7+1+1] = weather_17.loc[7*i + j,'기온']
                temp_sig_18[7*i + j][7+1+1+1] = weather_17.loc[7*i + j,'강수량']
                temp_sig_18[7*i + j][7+1+1+1+1] = weather_17.loc[7*i + j,'습도']
                temp_sig_18[7*i + j][7+1+1+1+1+1] = weather_17.loc[7*i + j,'적설량']
                
                if weekday == 'fri': # 금요일
                    temp_sig_18[7*i + j][7+1] = 1 # before holiday
                elif weekday == 'sat': # 토요일
                    temp_sig_18[7*i + j][7+1] = 1 # before holiday
                    temp_sig_18[7*i + j][7] = 1 # holiday
                elif weekday == 'sun': # 일요일
                    temp_sig_18[7*i + j][7] = 1 # holiday
        # 2019    
        for i in range(21): # 21 weeks + a
            for j, (weekday, val) in enumerate(weekday_dict_17.items()):
                temp_sig_19[7*i + j][val] = 1 # weekday
                temp_sig_19[7*i + j][7+1+1] = weather_17.loc[7*i + j,'기온']
                temp_sig_19[7*i + j][7+1+1+1] = weather_17.loc[7*i + j,'강수량']
                temp_sig_19[7*i + j][7+1+1+1+1] = weather_17.loc[7*i + j,'습도']
                temp_sig_19[7*i + j][7+1+1+1+1+1] = weather_17.loc[7*i + j,'적설량']
                
                if weekday == 'fri': # 금요일
                    temp_sig_19[7*i + j][7+1] = 1 # before holiday
                elif weekday == 'sat': # 토요일
                    temp_sig_19[7*i + j][7+1] = 1 # before holiday
                    temp_sig_19[7*i + j][7] = 1 # holiday
                elif weekday == 'sun': # 일요일
                    temp_sig_19[7*i + j][7] = 1 # holiday
        
        # the last days
        for i in range(4):
            temp_sig_17[21*7 + i][weekday_dict_17[weekday_list[i]]] = 1 # weekday
            temp_sig_17[21*7 + i][7+1+1] = weather_17.loc[21*7 + i,'기온']
            temp_sig_17[21*7 + i][7+1+1+1] = weather_17.loc[21*7 + i,'강수량']
            temp_sig_17[21*7 + i][7+1+1+1+1] = weather_17.loc[21*7 + i,'습도']
            temp_sig_17[21*7 + i][7+1+1+1+1+1] = weather_17.loc[21*7 + i,'적설량']
            temp_sig_18[21*7 + i][weekday_dict_18[weekday_list[i]]] = 1 # weekday
            temp_sig_18[21*7 + i][7+1+1] = weather_18.loc[21*7 + i,'기온']
            temp_sig_18[21*7 + i][7+1+1+1] = weather_18.loc[21*7 + i,'강수량']
            temp_sig_18[21*7 + i][7+1+1+1+1] = weather_18.loc[21*7 + i,'습도']
            temp_sig_18[21*7 + i][7+1+1+1+1+1] = weather_18.loc[21*7 + i,'적설량']
        for i in range(5):
            temp_sig_19[21*7 + i][weekday_dict_19[weekday_list[i]]] = 1 # weekday
            temp_sig_19[21*7 + i][7+1+1] = weather_19.loc[21*7 + i,'기온']
            temp_sig_19[21*7 + i][7+1+1+1] = weather_19.loc[21*7 + i,'강수량']
            temp_sig_19[21*7 + i][7+1+1+1+1] = weather_19.loc[21*7 + i,'습도']
            temp_sig_19[21*7 + i][7+1+1+1+1+1] = weather_19.loc[21*7 + i,'적설량']
        
        temp_sig = torch.cat([temp_sig_17, temp_sig_18, temp_sig_19], dim=0)

        train_temp_sig_inputs, test_temp_sig_inputs = [], []
        train_infect_inputs, test_infect_inputs = [], []
        for i in range(TRAIN_DAYS):
            train_temp_sig_inputs.append(temp_sig[i:i+LSTM_SEQUENCE_LENGTH])
            train_infect_inputs.append(seoul_flu[i:i+LSTM_SEQUENCE_LENGTH])
        for i in range(TRAIN_DAYS, TRAIN_DAYS + TEST_DAYS):
            test_temp_sig_inputs.append(temp_sig[i:i+LSTM_SEQUENCE_LENGTH])
            test_infect_inputs.append(seoul_flu[i:i+LSTM_SEQUENCE_LENGTH])
        
        # input concatenate
        print("=== concat ===")
        train_inputs = []
        for i in range(TRAIN_DAYS):
            train_g_user_inputs = []
            for j in range(LSTM_SEQUENCE_LENGTH):
                train_g_user_inputs.append([train_user_node_list[i][j], train_user_edge_inputs[i][j][0].edge_index, 
                                                    torch.tensor(train_user_edge_inputs[i][j][1].astype(float))])
            train_inputs.append([train_g_user_inputs, train_infect_inputs[i], train_temp_sig_inputs[i]])

        test_inputs = []
        for i in range(TEST_DAYS):
            test_g_user_inputs = []
            for j in range(LSTM_SEQUENCE_LENGTH):
                test_g_user_inputs.append([test_user_node_list[i][j], test_user_edge_inputs[i][j][0].edge_index, 
                                                   torch.tensor(test_user_edge_inputs[i][j][1].astype(float))])
            test_inputs.append([test_g_user_inputs, test_infect_inputs[i], test_temp_sig_inputs[i]])

        ###################
        ###### label ######
        
        print('== label ready ==')
        # user, infect
        # sty label
        node_seoul_user_reshape = node_seoul_user.reshape([-1, NUM_DONG])
        target_user_sty = []
        for i, ymd_time in enumerate(ymd_time_lst):
            target_user_sty.append(node_seoul_user_reshape[i])

        # mv_label                   
        mv_user = pd.DataFrame(edge_seoul_user, columns=['etl_ymd', 'from_dong', 'to_dong', 'mv_num'])
        user_to_dong_grouped = mv_user.groupby(['etl_ymd', 'to_dong']).sum()['mv_num']
        user_to_dong_grouped = user_to_dong_grouped.reset_index()
        
        target_user_mv = []
        for i, ymd_time in tqdm(enumerate(ymd_time_lst)):
            target_user_mv_dong = torch.zeros([NUM_DONG])
            for j, dong in enumerate(dong_lst):
                if not user_to_dong_grouped[(user_to_dong_grouped['etl_ymd']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)].empty:
                    target_user_mv_dong[j] = user_to_dong_grouped[(user_to_dong_grouped['etl_ymd']==ymd_time) & (user_to_dong_grouped['to_dong']==dong)]['mv_num'].item()
            target_user_mv.append(target_user_mv_dong)

        train_target, test_target = [], []
        for i in range((LSTM_SEQUENCE_LENGTH + DELAY_DAYS), (LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS)):
            train_target_user = target_user_sty[i] + target_user_mv[i]
            train_target_infect = seoul_flu[i]
            train_target.append([train_target_user, train_target_infect])

        for i in range((LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS), (LSTM_SEQUENCE_LENGTH + DELAY_DAYS + TRAIN_DAYS + TEST_DAYS)):
            test_target_user = target_user_sty[i] + target_user_mv[i]
            test_target_infect = seoul_flu[i]
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

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    print('=== data preprocess done ===')
    
    return train_dataset, train_loader, test_dataset, test_loader, min_max_values

if __name__ == '__main__':
    import easydict
    args = easydict.EasyDict()
    args['device'] = 'cuda:0'
    args['train_days'] = 296
    args['delay_days'] = 0
    args['test_days'] = 148
    args['num_epochs'] = 250
    args['learning_rate'] = 1e-5
    args['num_dong'] = 25
    args['lstm_num_layers'] = 1
    args['lstm_input_size'] = 50
    args['lstm_hidden_size'] = 100
    args['lstm_sequence_length'] = 7
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
    
    train_dataset, train_loader, test_dataset, test_loader, min_max_values = data_loader_flu(args)
    

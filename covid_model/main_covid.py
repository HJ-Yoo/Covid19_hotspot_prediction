from models_covid import GCN_LSTM_covid
from data_covid import data_loader_covid

import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 spatial temporal modeling')

    parser.add_argument('--device', default='cuda:0', type=str, help='which device to use')
     
    # data hyperparameter
    parser.add_argument('--train_days', default=14, type=int, help='# of training days')
    parser.add_argument('--delay_days', default=4, type=int, help='deflaut:0')
    parser.add_argument('--test_days', default=4, type=int, help='# of testing days')
    
    # model hyperparameters
    parser.add_argument('--num_epochs', default=30, type=int, help='# of epochse')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_dong', default=25, type=int, help='# of nodes, seoul admdong=432, seoul sgg=25, daegu admdong=139')
    parser.add_argument('--lstm_num_layers', default=1, type=int, help='# of lstm layers, default=1')
    parser.add_argument('--lstm_input_size', default=50, type=int, help='lstm input size, default=50')
    parser.add_argument('--lstm_hidden_size', default=100, type=int, help='lstm hidden size, default=100')
    parser.add_argument('--lstm_sequence_length', default=7, type=int, help='lstm sequence length, covid=1, flu=7')
    parser.add_argument('--graph_conv_feature_dim_user', default=20, type=int, help='gcn conv feature dim, default=20')
    parser.add_argument('--graph_conv_feature_dim_infect', default=20, type=int, help='gcn conv feature dim, default=20')
    
    # architecture switch
    parser.add_argument('--deep_graph', help='2 layers of gcn', action='store_true')
    parser.add_argument('--concat_fc', help='(g_user, g_infect, temp_sig) concat fc layer', action='store_true')
    parser.add_argument('--temp_all_concat', help='temp_all concat fc layer. without it, just lstm output = output', action='store_true')
    parser.add_argument('--all_concat', help='temp_all + lstm_inputs + lstm_output concat fc layer. without it, just lstm output = output', action='store_true')
    
    # Miscellaneous
    parser.add_argument('--loss', default='mse', type=str, help='which loss use to train, mse or rmlse')
    parser.add_argument('--task', default='real_feb', type=str, help='which task to train, real_feb or fake_full')
    parser.add_argument('--data_path', default='/home/hyungjun/jupyter/KT_covid19/data/', type=str, help='base path for preprocessed dataset')
    parser.add_argument('--data_normalize', help='data_normalize', action='store_true')
    parser.add_argument('--result_path', default='./results/', type=str, help='directory for results')
   
    args = parser.parse_args()

    return args
                  
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

    
def main(model, train_dataloader, test_dataloader, learning_rate, num_epochs, args, min_max_values=None):
    '''
    모델 initialization, train, validation 과 log와 학습모델 저장 등
    '''
    # otimizer : ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # learning rate scheculer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*args.num_epochs/10)
    # loss
    mse_criterion = nn.MSELoss()
    rmsle_criterion = RMSLELoss()
    
    model.to(args.device)
    model.train()
    
    # log 저장용
    train_log_pd = pd.DataFrame(np.zeros([len(train_dataloader)*num_epochs, 5]), columns=['iterations', 'train_user_mse', 'train_infect_mse', 'train_user_rmsle', 'train_infect_rmsle'])
    valid_log_pd = pd.DataFrame(np.zeros([len(test_dataloader)*num_epochs, 5]), columns=['iterations', 'valid_user_mse', 'valid_infect_mse', 'valid_user_rmsle', 'valid_infect_rmsle'])
    min_valid_log = 1e+10
    
    for epoch in tqdm(range(num_epochs)):
        with tqdm(train_dataloader, total=len(train_dataloader), leave=False) as pbar:
            for i, data in enumerate(pbar):
                train_input, train_target = data[0], data[1]
                # print('train_input : ',train_input)
                # print('train_target : ',train_target.shape)
                
                # train
                user_output, infect_output = model(train_input, task=args.task, data_normalize=args.data_normalize, min_max_values=min_max_values, deep_graph=args.deep_graph)
                user_mse = mse_criterion(user_output, train_target[0].type(torch.FloatTensor).to(args.device))
                infect_mse = mse_criterion(infect_output, train_target[1].type(torch.FloatTensor).to(args.device))
                user_rmsle = rmsle_criterion(user_output, train_target[0].type(torch.FloatTensor).to(args.device))
                infect_rmsle = rmsle_criterion(infect_output, train_target[1].type(torch.FloatTensor).to(args.device))
                
                if args.loss == 'mse':
                    loss = user_mse + infect_mse
                elif args.loss == 'rmsle':
                    loss = user_rmsle + infect_rmsle
                
                train_log_pd['iterations'][epoch*len(train_dataloader)+i] = epoch*len(train_dataloader)+i
                train_log_pd['train_user_mse'][epoch*len(train_dataloader)+i] = user_mse.item()
                train_log_pd['train_infect_mse'][epoch*len(train_dataloader)+i] = infect_mse.item()
                train_log_pd['train_user_rmsle'][epoch*len(train_dataloader)+i] = user_rmsle.item()
                train_log_pd['train_infect_rmsle'][epoch*len(train_dataloader)+i] = infect_rmsle.item()
                
                # log, model 저장명
                filename = os.path.join('covid_{}_lr{}_d{}-{}-{}_s{}_cat{}_temp{}_all{}_lstm-h-dim{}_lstm-in-dim{}_g-user-dim{}_g-infect-dim{}_deep{}'.format(
                                                                                                                    args.loss,
                                                                                                                    args.learning_rate, 
                                                                                                                    args.train_days,
                                                                                                                    args.delay_days, 
                                                                                                                    args.test_days,
                                                                                                                    args.lstm_sequence_length,
                                                                                                                    args.concat_fc,
                                                                                                                    args.temp_all_concat,
                                                                                                                    args.all_concat,
                                                                                                                    args.lstm_hidden_size,
                                                                                                                    args.lstm_input_size,
                                                                                                                    args.graph_conv_feature_dim_user,
                                                                                                                    args.graph_conv_feature_dim_infect,
                                                                                                                    args.deep_graph))
                                                           
                train_log_pd.to_csv(args.result_path + 'logs/train_' + filename + '.csv', index=False)

                pbar.set_description('loss=%g' % loss.item())
                
                # model update (backpropagation)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            # schedular update
            scheduler.step()
                
            # validation
            mean_valid_user_mse, mean_valid_infect_mse, mean_valid_user_rmsle, mean_valid_infect_rmsle = validation(model, test_dataloader,
                                                                                                                    task=args.task, 
                                                                                                                    device=args.device, 
                                                                                                                    data_normalize=args.data_normalize, 
                                                                                                                    min_max_values=min_max_values,
                                                                                                                    deep_graph=args.deep_graph)
            valid_log_pd['iterations'][epoch*len(test_dataloader)+i] = epoch*len(test_dataloader)+i
            valid_log_pd['valid_user_mse'][epoch*len(test_dataloader)+i] = mean_valid_user_mse
            valid_log_pd['valid_infect_mse'][epoch*len(test_dataloader)+i] = mean_valid_infect_mse
            valid_log_pd['valid_user_rmsle'][epoch*len(test_dataloader)+i] = mean_valid_user_rmsle
            valid_log_pd['valid_infect_rmsle'][epoch*len(test_dataloader)+i] = mean_valid_infect_rmsle
            valid_log_pd.to_csv(args.result_path + 'logs/valid_' + filename + '.csv', index=False)
            mean_mse_valid_mse = mean_valid_user_mse + mean_valid_infect_mse
            
            # best valid accrucy마다 모델 저장 
            if min_valid_log >= mean_mse_valid_mse:
                min_valid_log = mean_mse_valid_mse
                print('best valid update => epochs : {}, iter : {}, loss : {}'.format(epoch, i, min_valid_log))
                best_valid_model = model
                model_path = os.path.join(args.result_path+'models/' + filename)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                model_filename = os.path.join(model_path + '/epochs_{}_iter_{}.pt'.format(epoch, i))
                with open(model_filename, 'wb') as f:
                    state_dict = model.state_dict()
                    torch.save(state_dict, f)


def validation(model, dataloader, task, device, data_normalize, min_max_values, deep_graph):
    '''
    매 epoch이 끝날때마다 validation error를 측정하고 log 기록
    '''
    model.eval()
    mse_criterion = nn.MSELoss()
    rmsle_criterion = RMSLELoss()
    
    valid_user_mse, valid_infect_mse, valid_user_rmsle, valid_infect_rmsle = [], [], [], []
    with tqdm(dataloader, total=len(dataloader), leave=False) as pbar:
        for i, data in enumerate(pbar):
            test_input, test_target = data[0], data[1]
            user_output, infect_output = model(test_input, task=task, data_normalize=data_normalize, min_max_values=min_max_values, deep_graph=deep_graph)
            user_mse = mse_criterion(user_output, test_target[0].type(torch.FloatTensor).to(device))
            infect_mse = mse_criterion(infect_output, test_target[1].type(torch.FloatTensor).to(device))
            user_rmsle = rmsle_criterion(user_output, test_target[0].type(torch.FloatTensor).to(device))
            infect_rmsle = rmsle_criterion(infect_output, test_target[1].type(torch.FloatTensor).to(device))
            
            valid_user_mse.append(user_mse.item())
            valid_infect_mse.append(infect_mse.item())
            valid_user_rmsle.append(user_rmsle.item())
            valid_infect_rmsle.append(infect_rmsle.item())
            
    model.train()
    
    return np.mean(valid_user_mse), np.mean(valid_infect_mse), np.mean(valid_user_rmsle), np.mean(valid_infect_rmsle)
        
if __name__ == '__main__':
    # argument parser 실행
    args = parse_args()
    
    # dataset load
    train_dataset, train_loader, test_dataset, test_loader, min_max_values = data_loader_covid(args)
    
    # model load
    model = GCN_LSTM_covid(args)

    print('=== training start ===' )
    main(model, train_loader, test_loader, learning_rate=args.learning_rate, num_epochs=args.num_epochs, args=args, min_max_values = min_max_values)
    
from models import GC_TG_LSTM
from data import data_loader
# from data_new_idx import data_loader_new_idx

import argparse
import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 spatial temporal modeling')

    parser.add_argument('--device', default='cuda:0', type=str, help='which device to use')
     
    # data hyperparameter
    parser.add_argument('--train_days', default=14, type=int, help='# of epochse')
    parser.add_argument('--delay_days', default=4, type=int, help='# of epochse')
    parser.add_argument('--test_days', default=4, type=int, help='# of epochse')
    
    # model hyperparameters
    parser.add_argument('--num_epochs', default=30, type=int, help='# of epochse')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--num_dong', default=142, type=int, help='# of epochse')
    parser.add_argument('--lstm_num_layers', default=3, type=int, help='# of epochse')
    parser.add_argument('--lstm_input_size', default=30, type=int, help='# of epochse')
    parser.add_argument('--lstm_hidden_size', default=50, type=int, help='# of epochse')
    parser.add_argument('--lstm_sequence_length', default=7, type=int, help='# of epochse')
    parser.add_argument('--graph_conv_feature_dim_user', default=10, type=int, help='# of epochse')
    parser.add_argument('--graph_conv_feature_dim_infect', default=10, type=int, help='# of epochse')
    
    # architecture switch
    parser.add_argument('--concat_fc', help='(g_user, g_infect, temp_sig) concat fc layer', action='store_true')
    parser.add_argument('--temp_all_concat', help='temp_all concat fc layer. without it, just lstm output = output', action='store_true')
    parser.add_argument('--all_concat', help='temp_all + lstm_inputs + lstm_output concat fc layer. without it, just lstm output = output', action='store_true')
    
    # Miscellaneous
    parser.add_argument('--data_path', default='/home/hyungjun/jupyter/KT_covid19/data/', type=str, help='base path for preprocessed dataset')
    parser.add_argument('--result_path', default='./results/', type=str, help='directory for results')
#     parser.add_argument('--save_steps', default=100, type=int, help='model and data save steps')
#     parser.add_argument('--seed', default=2020, type=int, help='random seed')

    args = parser.parse_args()

    return args


def main(model, train_dataloader, test_dataloader, learning_rate, num_epochs, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*args.num_epochs/10)
    
    model.to(args.device)
    model.train()
    
    train_log_pd = pd.DataFrame(np.zeros([len(train_dataloader)*num_epochs, 2]), columns=['iterations', 'train_loss'])
    valid_log_pd = pd.DataFrame(np.zeros([len(test_dataloader)*num_epochs, 3]), columns=['iterations', 'mse_valid_loss', 'mae_valid_loss'])
    min_valid_log = 1e+2
    
    for epoch in tqdm(range(num_epochs)):
        with tqdm(train_dataloader, total=len(train_dataloader), leave=False) as pbar:
            for i, data in enumerate(pbar):
                train_input = data[0]
                #print('train_input : ',train_input)
                train_target = data[1].type(torch.FloatTensor).to(args.device)
                #print('train_target : ',train_target)

                # train
                output = model(train_input)
                loss = torch.nn.functional.mse_loss(output, train_target.squeeze(0), )
                
                train_log_pd['iterations'][epoch*len(train_dataloader)+i] = epoch*len(train_dataloader)+i
                train_log_pd['train_loss'][epoch*len(train_dataloader)+i] = loss.item()

                filename = os.path.join('lr{}_d{}-{}-{}_s{}_cat{}_temp{}_all{}_lstm-h-dim{}_lstm-in-dim{}_g-user-dim{}_g-infect-dim{}'.format(
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
                                                                                                                    args.graph_conv_feature_dim_infect))
                                                           
                train_log_pd.to_csv(args.result_path + 'train_' + filename + '.csv', index=False)

                pbar.set_description('loss=%g' % loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            scheduler.step()
                
            # validation
            mean_mse_valid_log, mean_mae_valid_log = validation(model, test_dataloader, args.device)
            valid_log_pd['iterations'][epoch*len(test_dataloader)+i] = epoch*len(test_dataloader)+i
            valid_log_pd['mse_valid_loss'][epoch*len(test_dataloader)+i] = mean_mse_valid_log
            valid_log_pd['mae_valid_loss'][epoch*len(test_dataloader)+i] = mean_mae_valid_log
            valid_log_pd.to_csv(args.result_path + 'valid_' + filename + '.csv', index=False)
            if min_valid_log >= mean_mse_valid_log:
                min_valid_log = mean_mse_valid_log
                print('best valid update => epochs : {}, iter : {}, loss : {}'.format(epoch, i, min_valid_log))
                best_valid_model = model

                model_path = os.path.join(args.result_path+'models/' + filename)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                model_filename = os.path.join(model_path + '/epochs_{}_iter_{}.pt'.format(epoch, i))
                with open(model_filename, 'wb') as f:
                    state_dict = model.state_dict()
                    torch.save(state_dict, f)

            

def validation(model, dataloader, device):
    model.eval()
    
    mse_valid_log, mae_valid_log = [], []
    with tqdm(dataloader, total=len(dataloader), leave=False) as pbar:
        for i, data in enumerate(pbar):
            test_input = data[0]
            test_target = data[1].type(torch.FloatTensor).to(device)
            
            output = model(test_input)
            mse_loss = torch.nn.functional.mse_loss(output, test_target.squeeze(0))
            mae_loss = torch.nn.functional.l1_loss(output, test_target.squeeze(0))
            mse_valid_log.append(mse_loss.item())
            mae_valid_log.append(mae_loss.item())
            
    model.train()
    
    return np.mean(mse_valid_log), np.mean(mae_valid_log)
        
if __name__ == '__main__':
    args = parse_args()
#     make_result_dir(args)
    train_dataset, train_loader, test_dataset, test_loader = data_loader(args)
    model = GC_TG_LSTM(args)

    print('=== training start ===' )
    main(model, train_loader, test_loader, learning_rate=args.learning_rate, num_epochs=args.num_epochs, args=args)
    
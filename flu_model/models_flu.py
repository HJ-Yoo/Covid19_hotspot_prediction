import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import networkx as nx

from torch_geometric.nn import GCNConv
#vfrom torch_geometric.nn import

def denormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

class GC_TG_LSTM_flu(nn.Module):
    def __init__(self, args):
        super(GC_TG_LSTM_flu, self).__init__()
        
        # hyperparameters
        self.device = args.device
        self.lstm_sequence_length = args.lstm_sequence_length
        self.lstm_input_size = args.lstm_input_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_num_layers = args.lstm_num_layers
        self.num_dong = args.num_dong
        # architecture switch
        self.temp_all_concat = args.temp_all_concat
        self.concat_fc = args.concat_fc
        self.all_concat = args.all_concat
        self.batch_size = args.batch_size
        
        # graph conv for user
        self.graph_conv_user1 = GCNConv(1,10)
        self.graph_act1_user1 = nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        self.graph_conv_user2 =  GCNConv(10, args.graph_conv_feature_dim_user)
        self.graph_act1_user2 =  nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        self.graph_fc_user = nn.Linear(self.num_dong * args.graph_conv_feature_dim_user, args.lstm_input_size)
        self.graph_act2_user = nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        
        # graph conv for infect
        self.infect_fc1 = nn.Linear(25, 128)
        self.infect_fc1_act = nn.ReLU(inplace=False)
        self.infect_fc2 = nn.Linear(128, 128)
        self.infect_fc2_act = nn.ReLU(inplace=False)
        self.infect_fc3 = nn.Linear(128, 64)
        self.infect_fc3_act = nn.ReLU(inplace=False)
        self.infect_fc4 = nn.Linear(64, args.lstm_input_size)
        self.infect_fc4_act = nn.ReLU(inplace=False)
                
        # lstm
        self.lstm = nn.LSTM(input_size=args.lstm_input_size, hidden_size=args.lstm_hidden_size, num_layers=args.lstm_num_layers, batch_first=True)
        self.lstm_act1 =  nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        self.lstm_fc = nn.Linear(args.lstm_hidden_size, self.num_dong*2)
        self.lstm_act2 = nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        
        # temp signal single day 
        self.temp_sig_wth_fc1 = nn.Linear(13, 64)
        self.temp_sig_wth_fc1_act = nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1)  # 
        self.temp_sig_wth_fc2 = nn.Linear(64, args.lstm_input_size)
        self.temp_sig_wth_fc2_act = nn.ReLU(inplace=False)
        
        # concatenate FC
        self.concat_fc = nn.Linear(args.lstm_input_size*3, args.lstm_input_size)
        self.concat_fc_act = nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
                
        # temp signal all
        self.temp_all_fc = nn.Linear(13*self.lstm_sequence_length, self.num_dong*2)
        self.temp_all_act =  nn.ReLU(inplace=False) # nn.LeakyReLU(negative_slope=0.1) #
        self.temp_all_output_fc = nn.Linear(args.num_dong*2, args.num_dong*2)

        self.all_cat_fc = nn.Linear(self.lstm_sequence_length, args.num_dong)
        self.all_cat_act = nn.ReLU(inplace=False)
        self.all_cat_output_fc = nn.Linear(args.num_dong*3, self.num_dong*2)
        
    def forward(self, batch, data_normalize, min_max_values=None, deep_graph=None):
        # data decomposition
        '''
        len(data[0]) = 3 : user / infect / temp
        len(data[0][0]) = 168 : user_graph_list 
        len(data[0][0][0]) = 3 : user_graph 하나
        len(train_inputs[0][0][0][0]) # user_graph_node
        '''
        self.lstm_input_tensor = torch.zeros([self.batch_size , self.lstm_sequence_length, self.lstm_input_size]).to(self.device)
        self.lstm_output_tensor =  torch.zeros([self.batch_size , self.num_dong*2]).to(self.device)
        for j, data in enumerate(batch):
            user_graph_list, infect_list, temp_sig_list = data[0], data[1], data[2] # [168, [ [node], [edge_list], [edge_weight] ] ]
            # 각 time 하나씩 
            for i in range(self.lstm_sequence_length): 
                graph_user = self.graph_conv_user1(user_graph_list[i][0].unsqueeze(1).type(torch.FloatTensor).to(self.device), 
                                                   user_graph_list[i][1].to(self.device), user_graph_list[i][2].to(self.device))
                graph_user = self.graph_act1_user1(graph_user)
                if deep_graph:
                    graph_user = self.graph_conv_user2(graph_user.reshape([self.num_dong, -1]).type(torch.FloatTensor).to(self.device),
                                                       user_graph_list[i][1].to(self.device), user_graph_list[i][2].to(self.device))
                    graph_user = self.graph_act1_user2(graph_user)
                graph_user = self.graph_fc_user(graph_user.view(-1).type(torch.FloatTensor).to(self.device))
                graph_user = self.graph_act2_user(graph_user)
                
                infect = self.infect_fc1_act(self.infect_fc1(infect_list[i].type(torch.FloatTensor).to(self.device)))
                infect = self.infect_fc2_act(self.infect_fc2(infect))
                infect = self.infect_fc3_act(self.infect_fc3(infect))
                infect = self.infect_fc4_act(self.infect_fc4(infect))

                temp_sig = self.temp_sig_wth_fc1_act(self.temp_sig_wth_fc1(temp_sig_list[i].type(torch.FloatTensor).to(self.device)))
                temp_sig = self.temp_sig_wth_fc2_act(self.temp_sig_wth_fc2(temp_sig))
                # print('temp_sig : ',temp_sig)

                if self.concat_fc:
                    lstm_input = self.concat_fc(torch.cat((graph_user, infect, temp_sig)))
                    lstm_input = self.concat_fc_act(lstm_input)
                else:
                    lstm_input = graph_user + graph_infect + temp_sig
                # print('lstm_input : ',lstm_input)
                self.lstm_input_tensor[j][i] = lstm_input

            h_0 = Variable(torch.randn([1, self.batch_size, self.lstm_hidden_size], requires_grad=True)).to(self.device)
            c_0 = Variable(torch.randn([1, self.batch_size, self.lstm_hidden_size], requires_grad=True)).to(self.device)
            lstm_output, (h_out, c_out) = self.lstm(self.lstm_input_tensor, (h_0, c_0))
            # print('lstm_output : ', lstm_output)
        
            lstm_output = self.lstm_act1(lstm_output)
            
            for i in range(self.batch_size):
                self.lstm_output_tensor[i] = self.lstm_act2(self.lstm_fc(lstm_output[i][-1].view(-1)))
            # print('output : ', lstm_output)
        
        output = self.lstm_output_tensor
        
        if self.temp_all_concat:
            temp_sig_all = self.temp_all_fc(temp_sig_list.view(-1).to(self.device))
            temp_sig_all = self.temp_all_act(temp_sig_all.view(-1).to(self.device))
            
            output = self.temp_all_output_fc(torch.cat((lstm_output, temp_sig_all)))
        
        if self.all_concat:
            temp_sig_all = self.temp_all_fc(temp_sig_list.view(-1).to(self.device))
            temp_sig_all = self.temp_all_act(temp_sig_all.view(-1).to(self.device))
            
            lstm_input_all = self.all_cat_fc(lstm_inputs.view(-1))
            lstm_input_all = self.all_cat_act(lstm_input_all)
            
            output = self.all_cat_output_fc(torch.cat((lstm_output, lstm_input_all, temp_sig_all)))
            
        user_pred, infect_pred = output.view(-1)[:self.num_dong*self.batch_size].reshape([self.batch_size,-1]), output.view(-1)[self.num_dong*self.batch_size:].reshape([self.batch_size,-1])
        
        if data_normalize:
            for i in range(len(user_pred)):
                user_pred[i] = denormalize(user_pred[i], min_max_values[0]+min_max_values[2], min_max_values[1]+min_max_values[3])
                
        return user_pred, infect_pred

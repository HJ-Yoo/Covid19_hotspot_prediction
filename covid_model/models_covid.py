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

class GCN_LSTM_covid(nn.Module):
    def __init__(self, args):
        super(GCN_LSTM_covid, self).__init__()
        
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
        
        # graph conv for user
        self.graph_conv_user1 = GCNConv(1,10)
        self.graph_act1_user1 = nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        self.graph_conv_user2 =  GCNConv(10, args.graph_conv_feature_dim_user)
        self.graph_act1_user2 =  nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        
        self.graph_fc_user = nn.Linear(self.num_dong * args.graph_conv_feature_dim_user, args.lstm_input_size)
        self.graph_act2_user = nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        # self.graph_fc_user2 = nn.Linear()
        # sefl.graph_act2_user2 = nn.ReLU()
        
        # graph conv for infect
        self.graph_conv_infect1 = GCNConv(1, 10)
        self.graph_act1_infect1 =  nn.ReLU()  # nn.LeakyReLU(negative_slope=0.1) #
        self.graph_conv_infect2 = GCNConv(10, args.graph_conv_feature_dim_infect)
        self.graph_act1_infect2 =  nn.ReLU()  # nn.LeakyReLU(negative_slope=0.1) #
        self.graph_fc_infect = nn.Linear(self.num_dong * args.graph_conv_feature_dim_infect, args.lstm_input_size)
        self.graph_act2_infect = nn.ReLU() #  nn.LeakyReLU(negative_slope=0.1) #
        
        # test fc layer
        self.test_fc_user = nn.Linear(self.num_dong, args.lstm_input_size)
        self.test_fc_infect = nn.Linear(self.num_dong, args.lstm_input_size)
        
        # concatenate FC
        self.concat_fc = nn.Linear(args.lstm_input_size*3, args.lstm_input_size)
        self.concat_fc_act = nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        
        # lstm
        
        self.lstm = nn.LSTM(input_size=args.lstm_input_size, hidden_size=args.lstm_hidden_size, num_layers=args.lstm_num_layers)
        self.lstm_act1 =  nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        self.lstm_fc = nn.Linear(args.lstm_hidden_size, self.num_dong*2)
        self.lstm_act2 = nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        
        # temp signal single day 
        self.temp_single_fc = nn.Linear(57, args.lstm_input_size)
        self.temp_single_act = nn.ReLU() # nn.LeakyReLU(negative_slope=0.1)  # 
        
        # temp signal all
        self.temp_all_fc = nn.Linear(57*48*self.lstm_sequence_length, self.num_dong*2)
        self.temp_all_act =  nn.ReLU() # nn.LeakyReLU(negative_slope=0.1) #
        self.temp_all_output_fc = nn.Linear(args.num_dong*2, args.num_dong*2)

        self.all_cat_fc = nn.Linear(48*self.lstm_sequence_length*args.lstm_input_size, args.num_dong)
        self.all_cat_act = nn.ReLU()
        self.all_cat_output_fc = nn.Linear(args.num_dong*3, self.num_dong*2)
        
    def forward(self, x, task, data_normalize, min_max_values=None, deep_graph=None):
        # data decomposition
        '''
        len(data[0]) = 3 : user / infect / temp
        len(data[0][0]) = 168 : user_graph_list 
        len(data[0][0][0]) = 3 : user_graph 하나
        len(train_inputs[0][0][0][0]) # user_graph_node
        '''
        
        user_graph_list, infect_graph_list, temp_sig_list = x[0], x[1], x[2].squeeze(0) # [168, [ [node], [edge_list], [edge_weight] ] ]
        
        self.lstm_input_tensor = torch.zeros([1, 48*self.lstm_sequence_length, self.lstm_input_size]).to(self.device)
        
        for i in range(48*self.lstm_sequence_length):
            
            graph_user = self.graph_conv_user1(user_graph_list[i][0].reshape([self.num_dong,-1]).type(torch.FloatTensor).to(self.device), 
                                              user_graph_list[i][1].squeeze(0).to(self.device), user_graph_list[i][2].squeeze(0).to(self.device))
            graph_user = self.graph_act1_user1(graph_user)
            if deep_graph:
                graph_user = self.graph_conv_user2(graph_user.type(torch.FloatTensor).reshape([self.num_dong,-1]).to(self.device), 
                                                   user_graph_list[i][1].squeeze(0).to(self.device), user_graph_list[i][2].squeeze(0).to(self.device))
                graph_user = self.graph_act1_user2(graph_user)
            graph_user = self.graph_fc_user(graph_user.view(-1).type(torch.FloatTensor).to(self.device))
            graph_user = self.graph_act2_user(graph_user)
            # print('graph_user : ', graph_user)
            # print(self.graph_fc_user.bias.data)
    
            graph_infect = self.graph_conv_infect1(infect_graph_list[i][0].reshape([self.num_dong,-1]).type(torch.FloatTensor).to(self.device), 
                                                  infect_graph_list[i][1].squeeze(0).to(self.device), infect_graph_list[i][2].squeeze(0).to(self.device))
            graph_infect = self.graph_act1_infect1(graph_infect)
            if deep_graph:
                graph_infect = self.graph_conv_infect2(graph_infect.type(torch.FloatTensor).reshape([self.num_dong,-1]).to(self.device), 
                                                       infect_graph_list[i][1].squeeze(0).to(self.device), infect_graph_list[i][2].squeeze(0).to(self.device))
                graph_infect = self.graph_act1_infect2(graph_infect)
            graph_infect = self.graph_fc_infect(graph_infect.view(-1).type(torch.FloatTensor).to(self.device))
            graph_infect = self.graph_act2_infect(graph_infect)
            # print('graph_infect: ', graph_infect)
    
            temp_sig = self.temp_single_fc(temp_sig_list[i].to(self.device))
            temp_sig = self.temp_single_act(temp_sig)
            # print('temp_sig : ',temp_sig)
            
            if self.concat_fc:
                lstm_input = self.concat_fc(torch.cat((graph_user, graph_infect, temp_sig)))
                lstm_input = self.concat_fc_act(lstm_input)
            else:
                lstm_input = graph_user + graph_infect + temp_sig
            
            # print('lstm_input : ',lstm_input)
            self.lstm_input_tensor[0][i] = lstm_input
       
        h_0 = Variable(torch.randn([self.lstm_num_layers, 48*self.lstm_sequence_length, self.lstm_hidden_size], requires_grad=True)).to(self.device)
        c_0 = Variable(torch.randn([self.lstm_num_layers, 48*self.lstm_sequence_length, self.lstm_hidden_size], requires_grad=True)).to(self.device)
        lstm_output, (h_out, c_out) = self.lstm(self.lstm_input_tensor, (h_0, c_0))
        # print('lstm_output : ', lstm_output)
        
        lstm_output = self.lstm_act1(lstm_output[0][-1])
        lstm_output = self.lstm_fc(lstm_output.view(-1))
        lstm_output = self.lstm_act2(lstm_output)
        # print('output : ', lstm_output)
        
        output = lstm_output
        
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
            
        user_pred, infect_pred = output[:self.num_dong], output[self.num_dong:]
        if data_normalize:
            if task == 'real_feb':
                user_pred = denormalize(user_pred, min_max_values[0]+min_max_values[2], min_max_values[1]+min_max_values[3])
                infect_pred = denormalize(infect_pred, min_max_values[4]+min_max_values[6], min_max_values[5]+min_max_values[7])
            elif task == 'fake_full':
                user_pred = denormalize(user_pred, min_max_values[8]+min_max_values[10], min_max_values[9]+min_max_values[11])
                infect_pred = denormalize(infect_pred, min_max_values[12]+min_max_values[14], min_max_values[13]+min_max_values[15])
        
        return user_pred, infect_pred

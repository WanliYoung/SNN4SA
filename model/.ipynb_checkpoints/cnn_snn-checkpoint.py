import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self, snn_lstm_model, cnn_model, args):
        super(CombinedModel, self).__init__()
        self.snn_lstm = snn_lstm_model
        self.cnn = cnn_model
        
        # 全连接层
        fc_input_dim = 2 * args.label_num
        self.fc_combined = nn.Linear(fc_input_dim, args.label_num)

    def forward(self, x):
        with torch.no_grad():
            # 并行，分别通过snn_lstm和cnn处理
            spks, spk2, mem2 = self.snn_lstm(x)
            cnn_output = self.cnn(x)
            # print(f"x shape: {x.shape}")
            # print(f"mem2 shape: {mem2.shape}")
        
        combined_out = torch.cat((mem2, cnn_output), dim=1)
        
        # 通过额外的全连接层
        output = self.fc_combined(combined_out)
        return output
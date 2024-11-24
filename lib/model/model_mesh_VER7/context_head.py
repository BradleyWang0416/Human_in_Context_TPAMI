import torch.nn as nn


class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_in=3, dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # self.pre_fc = nn.Linear(dim_in, dim_rep)
        # self.pre_bn = nn.BatchNorm1d(dim_rep, momentum=0.1)
        # self.pre_relu = nn.ReLU(inplace=True)

    def forward(self, feat):
        '''
            Input: (N,T,J,C)-mean->(N,J,C)-reshape->(N,JC)-Linear->(N,hidden_dim)
        '''
        # feat = self.pre_fc(feat)
        # feat = feat.reshape(N, -1)
        # feat = self.pre_bn(feat)
        # feat = feat.reshape(N, T, J, C)
        # feat = self.pre_relu(feat)
        
        N, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 3, 1)         # (N, T, J, C) -> (N, J, C, T)
        feat = feat.mean(dim=-1)                # (N, J, C)
        feat = feat.reshape(N, -1)              # (N, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat


def append_context_head(model_instance):
    return
    # model_instance is an instance of a model class
    setattr(model_instance, 'context_head', ActionHeadClassification(dropout_ratio=0.5, dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048))
    original_forward = model_instance.forward

    def new_forward(*args, **kwargs):
        original_output = original_forward(*args, **kwargs)
        if isinstance(original_output, tuple):
            action_output = model_instance.context_head(original_output[0])
        elif isinstance(original_output, dict):
            action_output = model_instance.context_head(original_output['pred'])
        context_output = model_instance.context_head(original_output)
        return action_output

    model_instance.forward = new_forward
    print('Context head appended to the model')
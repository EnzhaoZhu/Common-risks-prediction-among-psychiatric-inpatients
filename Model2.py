from torch import nn
from transformers import XLNetModel
from transformers.modeling_utils import SequenceSummary
from config import Config
import torch

class Model2(nn.Module):
    """" Language model with multimodal design """

    def __init__(self):
        super(Model2, self).__init__()
        self.config = Config()
        self.xn = XLNetModel.from_pretrained(self.config.args.pretrain_path)
        self.Sequencesummary = SequenceSummary(self.xn.config)
        self.dropout = nn.Dropout(self.config.args.dropout_prob)

        self.fn1 = nn.Linear(self.config.args.input_numeric_size, self.config.args.input_numeric_size)
        self.fn2 = nn.Linear(self.config.args.hidden_size + self.config.args.input_numeric_size, self.config.args.mlp_nodes)
        self.fn3 = nn.Linear(self.config.args.mlp_nodes, self.config.args.mlp_nodes)

        self.clf = nn.Linear(self.config.args.mlp_nodes, self.config.args.num_class)

        for p in self.xn.parameters():
            p.requires_grad = True
        self.to(self.config.args.device)

    def forward(self, input_ids, attention_mask, token_type_ids, numbers):
        input_ids, attention_mask, token_type_ids = input_ids.to(self.config.args.device), \
            attention_mask.to(self.config.args.device), token_type_ids.to(self.config.args.device)
        x = self.xn(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        x = self.Sequencesummary(x.last_hidden_state)
        x = self.dropout(x)

        x1 = self.fn1(numbers)
        x1 = self.dropout(x1)

        x2 = torch.cat((x, x1), dim=1)   # Concat text feature and structured data feature
        x2 = self.fn2(x2)
        x2 = self.dropout(x2)
        x2 = self.fn3(x2)
        x2 = self.clf(x2)
        return x2
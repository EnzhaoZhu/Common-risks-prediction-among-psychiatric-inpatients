

from torch import nn
from transformers import XLNetModel
from transformers.modeling_utils import SequenceSummary
from config import Config


class Model1a(nn.Module):
    """ Multitask Language model, including Model 1a and Model 3a
        Model 1a, Multitask Language model with first medical record only
        Model 3a, Multitask Language model with semi-structured textualization design """

    def __init__(self):
        super(Model1a, self).__init__()
        self.config = Config()
        self.xn = XLNetModel.from_pretrained(self.config.args.pretrain_path)
        self.Sequencesummary = SequenceSummary(self.xn.config)
        self.dropout = nn.Dropout(self.config.args.dropout_prob)
        self.fn1 = nn.Linear(self.config.args.hidden_size, self.config.args.mlp_nodes)
        self.fn2 = nn.Linear(self.config.args.mlp_nodes, self.config.args.mlp_nodes)

        self.clf1 = nn.Linear(self.config.args.mlp_nodes, self.config.args.num_class)
        self.clf2 = nn.Linear(self.config.args.mlp_nodes, self.config.args.num_class)
        self.clf3 = nn.Linear(self.config.args.mlp_nodes, self.config.args.num_class)

        for p in self.xn.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids, attention_mask, token_type_ids = input_ids.to(self.config.args.device), \
            attention_mask.to(self.config.args.device), token_type_ids.to(self.config.args.device)
        x = self.xn(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        x = self.Sequencesummary(x.last_hidden_state)
        x = self.dropout(x)
        x = self.fn1(x)
        x = self.dropout(x)
        x = self.fn2(x)

        logits1 = self.clf1(x)
        logits2 = self.clf2(x)
        logits3 = self.clf3(x)
        return logits1, logits2, logits3






from torch import nn
from transformers import XLNetModel
from transformers.modeling_utils import SequenceSummary
from config import Config


class Model1(nn.Module):
    """ Language model, including Model 1 and Model 3
        Model 1, language model with first medical record only
        Model 3, language model with semi-structured textualization design """

    def __init__(self):
        super(Model1, self).__init__()
        self.config = Config()  # Create config instance
        self.xn = XLNetModel.from_pretrained(self.config.args.pretrain_path)
        self.Sequencesummary = SequenceSummary(self.xn.config)
        self.dropout = nn.Dropout(self.config.args.dropout_prob)
        self.fn1 = nn.Linear(self.config.args.hidden_size, self.config.args.mlp_nodes)
        self.fn2 = nn.Linear(self.config.args.mlp_nodes, self.config.args.mlp_nodes)

        self.clf = nn.Linear(self.config.args.mlp_nodes, self.config.args.num_class)

        # Ensure XLNet model parameters are trainable
        for p in self.xn.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids, attention_mask, token_type_ids = input_ids.to(self.config.args.device), \
            attention_mask.to(self.config.args.device), token_type_ids.to(self.config.args.device)
        x = self.xn(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        x = self.Sequencesummary(x.last_hidden_state)
        x = self.dropout(x)
        x = self.fn1(x)  # Ensure to use the correct layer
        x = self.dropout(x)
        x = self.fn2(x)
        x = self.clf(x)
        return x




import argparse
import torch
from transformers import AutoTokenizer


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Model configuration")

        # Model arguments
        self.parser.add_argument('--pretrain_path', type=str, default='params/pretrain/xlnet_base',
                                 help="Path to the pre-trained model")
        self.parser.add_argument('--max_length', type=int, default=1200,
                                 help="Maximum length of input sequences")
        self.parser.add_argument('--dropout_prob', type=float, default=0.2,
                                 help="Dropout probability for regularization")
        self.parser.add_argument('--mlp_nodes', type=int, default=512,
                                 help="Number of nodes in the MLP layers")
        self.parser.add_argument('--hidden_size', type=int, default=768,
                                 help="Hidden size of the transformer model")
        self.parser.add_argument('--num_class', type=int, default=2,
                                 help="Number of classes per task")
        self.parser.add_argument('--input_numeric_size', type=int, default=139,
                                 help="Size of input numeric features")

        # Training arguments
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help="Batch size for training")
        self.parser.add_argument('--epoch_num', type=int, default=10,
                                 help="Number of training epochs")
        self.parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                                 help="Device to use for training ('cuda' or 'cpu')")
        self.parser.add_argument('--lr', type=float, default=1e-5,
                                 help="Learning rate for the optimizer")


        self.args = self.parser.parse_args()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_path)

    def __str__(self):
        config_str = "\nConfiguration Details:"
        for arg in vars(self.args):
            config_str += f"\n{arg}: {getattr(self.args, arg)}"
        return config_str

















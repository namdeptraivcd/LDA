import os
import torch
import argparse

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    JAR_DIR=os.path.join('data','evaluation')
    WIKI_DIR_KAGGLE=os.path.join('kaggle','input','wikipedia','wikipedia_bd')
    WIKI_DIR_COLAB=os.path.join('..','wikipedia','wikipedia_bd')
    WIKI_DIR_LOCAL=os.path.join('data','evaluation','wikipedia','wikipedia_bd')

    @staticmethod
    def create_new_parser():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def add_train_argument(parser):
        parser.add_argument('--track_loss',type=int,default=0)
        parser.add_argument('--hidden_size', type=int, default=256,
                            help='number of hidden units for hidden layers')
        parser.add_argument('--num_topics', type=int, default=50,
                            help='number of topics')
        parser.add_argument('--num_top_words',type=int,default=15)
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='dropout applied to layers')
        parser.add_argument('--use_lognormal', type=int, default=0,
                            help='Use LogNormal to approximate Dirichlet (flag sets True)')
        parser.add_argument('--num_epochs', type=int, default=48,
                            help='maximum training epochs')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
        parser.add_argument('--wd', type=float, default=0.0,
                            help='weight decay used for regularization')
        parser.add_argument('--epoch_size', type=int, default=2000,
                            help='number of training steps in an epoch')
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
        parser.add_argument('--checkpoint_dir',type=str,default=os.path.join('data','output_models'))
        parser.add_argument('--checkpoint_path',type=str,default='')
        parser.add_argument('--use_kaggle',type=int,default=0)
        parser.add_argument('--use_colab',type=int,default=0)
    @staticmethod
    def add_data_argument(parser):
        parser.add_argument('--data_dir',type=str,default=os.path.join('data','datasets'))
        parser.add_argument('--dataset',type=str,default='20NG')
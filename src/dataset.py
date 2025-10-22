import os
import pickle
import scipy
import numpy as np
from src.utils import Utils
from src.config import Config
class Dataset:
    def __init__(self,args):
        self.args=args
        self.dataset_dir=os.path.join(args.data_dir,args.dataset)
        self.load_dataset(self.dataset_dir,args.read_labels)
        self.vocab_size=len(self.vocab)
        self.plm_model=Config.PLM_MODEL
        # Just for simplication:
        n_train=int(0.8*self.train_bow.shape[0])
        self.train_data=self.train_bow[:n_train]
        self.valid_data=self.train_bow[n_train:]
        self.test_data=self.test_bow
        print("===> train_size: ", self.train_data.shape[0])
        print("===> valid_size: ", self.valid_data.shape[0])
        print("===> test_size: ", self.test_data.shape[0])
        print("===> vocab_size: ", self.vocab_size)
    def load_dataset(self,dataset_dir,read_labels):
        self.train_bow=scipy.sparse.load_npz(os.path.join(dataset_dir,'train_bow.npz')).toarray().astype('float32')
        self.test_bow=scipy.sparse.load_npz(os.path.join(dataset_dir,'test_bow.npz')).toarray().astype('float32')
        self.pretrained_WE=scipy.sparse.load_npz(os.path.join(dataset_dir,'word_embeddings.npz')).toarray().astype('float32')
        self.train_texts=Utils.read_file_txt(os.path.join(dataset_dir,'train_texts.txt'))
        self.test_texts=Utils.read_file_txt(os.path.join(dataset_dir,'test_texts.txt'))
        self.vocab_txt=Utils.read_file_txt(os.path.join(dataset_dir,'vocab.txt'))
        self.vocab={w:i for i,w in enumerate(self.vocab_txt)}
        if read_labels:
            self.train_labels=np.loadtxt(os.path.join(dataset_dir,'train_labels.txt'),dtype=int)
            self.test_labels=np.loadtxt(os.path.join(dataset_dir,'test_labels.txt'),dtype=int)
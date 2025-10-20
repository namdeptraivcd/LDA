import os
import pickle
import numpy as np
class Dataset:
    def __init__(self,data_dir,dataset):
        vocab_path=os.path.join(data_dir,dataset,'vocab.pkl')
        train_data_path=os.path.join(data_dir,dataset,'train.npy')
        test_data_path=os.path.join(data_dir,dataset,'test.npy')
        valid_data_path=os.path.join(data_dir,dataset,'valid.npy')

        with open(vocab_path,'rb') as f:
            self.vocab=pickle.load(f)
        
        def load_split(file_path):
            data = np.load(file_path, encoding='bytes')
            return np.array([
                np.bincount(x.astype('int'), minlength=len(self.vocab))
                for x in data if x.sum() > 0
            ])
        
        self.train_data=load_split(train_data_path)
        self.test_data=load_split(test_data_path)
        self.valid_data=load_split(valid_data_path)
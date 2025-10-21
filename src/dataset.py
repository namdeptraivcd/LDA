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
            self.vocab=pickle.load(f) # dict{word: index}
        
        def load_split(file_path):
            data = np.load(file_path, allow_pickle=True,encoding='latin1')
            res=[]
            for x in data:
                if x is not None and len(x)>0:
                    x_filtered=[val for val in x if val is not None]
                    if len(x_filtered)>0:
                        bincount=np.bincount(np.array(x_filtered).astype('int'),minlength=len(self.vocab))
                        if bincount.sum()>0:
                            res.append(bincount)
            if res:
                return np.array(res)
            else:
                raise ValueError('Data is empty')
        
        self.train_data=load_split(train_data_path)
        self.test_data=load_split(test_data_path)
        self.valid_data=load_split(valid_data_path)
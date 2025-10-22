import os
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence
from src.metrics import Metrics
from src.config import Config
class Trainer:
    def __init__(self,model,optimizer,args,dataset):
        self.model=model
        self.optimizer=optimizer
        self.args=args
        self.dataset=dataset
        self.device=Config.DEVICE
        self.batch_size=args.batch_size
        self.checkpoint_dir=args.checkpoint_dir
    def fit(self,train_data,valid_data,num_epochs):
        best_loss=float('inf')
        best_model=None
        for epoch in tqdm(range(num_epochs),desc='Training epochs'):
            self.train_epoch(train_data,epoch,num_epochs)
            val_loss=self.evaluate(valid_data)
            if val_loss<best_loss:
                # Update best model
                best_model={
                    'epoch':epoch,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                }
            if epoch >= 100 and epoch%100==0:
                self.save_checkpoint(epoch)
        best_checkpoint_path=self.save_best_model(best_model)
        self.load_checkpoint(best_checkpoint_path)
    def train_epoch(self,train_data,epoch,num_epochs):
        self.model.train()
        num_batches=len(train_data)//self.batch_size+(1 if len(train_data)%self.batch_size != 0 else 0)
        with tqdm(total=num_batches,desc=f'Epoch {epoch+1}/{num_epochs}',disable=not self.args.track_loss) as pbar:
            for start_idx in range(0,len(train_data),self.batch_size):
                end_idx=min(start_idx+self.batch_size,len(train_data))
                batch_train_data = torch.tensor(train_data[start_idx:end_idx],dtype=torch.float32).to(self.device)
                loss=self.get_loss(batch_train_data)
                # Remove old gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Update parameters
                self.optimizer.step()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
    def get_loss(self,inputs): 
        outputs,posterior=self.model(inputs) # @QUESTION: what is posterior?
        recon_loss=-torch.sum(inputs*outputs) # Reconstruction loss # @TODO: search this formula in paper
        if isinstance(posterior, LogNormal):
            prior = LogNormal(torch.zeros_like(posterior.loc), 
                            torch.ones_like(posterior.scale))
        else:  # Dirichlet
            prior = Dirichlet(torch.ones_like(posterior.concentration))
        kl_loss=torch.sum(kl_divergence(posterior, prior).to(self.device))
        loss=recon_loss+kl_loss
        return loss 
    def evaluate(self,valid_data):
        self.model.eval()
        with torch.no_grad():
            val_loss=0.0
            cnt=0
            for start_idx in range(0,len(valid_data),self.batch_size):
                end_idx=min(start_idx+self.batch_size,len(valid_data))
                valid_data_batch = torch.tensor(valid_data[start_idx:end_idx],dtype=torch.float32).to(self.device)
                loss=self.get_loss(valid_data_batch)
                val_loss+=loss
                cnt+=1
            val_loss/=cnt
        return val_loss
    def test(self,train_data,test_data,vocab,num_top_words):
        train_theta=self.get_theta(train_data)
        test_theta=self.get_theta(test_data)
        top_words_list_str=self.get_top_words(vocab,num_top_words)
        top_words_path=os.path.join(self.checkpoint_dir,f'top_words_{num_top_words}.txt')
        os.makedirs(self.checkpoint_dir,exist_ok=True)
        with open(top_words_path,'w') as f:
            for top_words_str in top_words_list_str:
                f.write(top_words_str+'\n')
        # Top words list
        topic=0
        for top_words_str in top_words_list_str:
            print(f'Topic {topic}: {top_words_str}')
            topic+=1
        # TD
        td=Metrics.td(top_words_list_str=top_words_list_str)
        print(f'TD_{num_top_words}: {td:.4f}')
        # Clustering and Classification
        if self.args.read_labels:
            # Clustering:
            # NMI
            nmi=Metrics.nmi(test_theta, self.dataset.test_labels)
            print(f'NMI: {nmi:.4f}')
            # Purity
            purity=Metrics.purity(test_theta, self.dataset.test_labels)
            print(f'Purity: {purity:.4f}')
            # Classification:
            # Accuracy
            acc=Metrics.accuracy(train_theta, test_theta, self.dataset.train_labels, self.dataset.test_labels, tune=self.args.tune_SVM)
            print(f'Accuracy: {acc:.4f}')
            # Marco-F1
            marco_f1=Metrics.accuracy(train_theta, test_theta, self.dataset.train_labels, self.dataset.test_labels, tune=self.args.tune_SVM)
            print(f'Marco-F1: {marco_f1:.4f}')
        # TC on Wiki
        tc_list,tc=Metrics.tc_on_wiki(use_kaggle=self.args.use_kaggle,use_colab=self.args.use_colab,top_words_path=top_words_path)
        print(f'TC_{num_top_words}: {tc:.4f}')
                
    def get_theta(self,data):
        """This method return a full version of theta, not batched theta

        Args:
            data (_type_): _description_
            
        Returns:
            _type_: _description_
        """
        data_size=data.shape[0]
        theta=[]
        with torch.no_grad():
            self.model.eval()
            for start_idx in range(0,data_size,self.batch_size):
                end_idx=min(start_idx+self.batch_size,data_size)
                batch_data=torch.tensor(data[start_idx:end_idx],dtype=torch.float32).to(self.device)
                batch_theta=self.model.get_theta(batch_data)
                theta.extend(batch_theta)
        return np.array(theta)
    def get_top_words(self,vocab,num_top_words) -> List[str]:
        beta=self.model.get_beta()
        idx2word={idx:word for word,idx in vocab.items()}
        top_words=[]
        for topic in beta:
            top_indices=np.argsort(topic)[-num_top_words:][::-1] # [::-1] is to reverse the array
            words=[idx2word[i] for i in top_indices]
            top_words.append(' '.join(words))
        return top_words
    def save_checkpoint(self,epoch):
        checkpoint={
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()
        }
        os.makedirs(self.checkpoint_dir,exist_ok=True)
        checkpoint_path=os.path.join(self.checkpoint_dir,f'checkpoint_{epoch}.pth')
        torch.save(checkpoint,checkpoint_path)
    def load_checkpoint(self,checkpoint_path):
        checkpoint=torch.load(checkpoint_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']+1
        return start_epoch
    def save_best_model(self,model):
        checkpoint=model
        os.makedirs(self.checkpoint_dir,exist_ok=True)
        checkpoint_path=os.path.join(self.checkpoint_dir,f'best_checkpoint.pth')
        torch.save(checkpoint,checkpoint_path)
        return checkpoint_path
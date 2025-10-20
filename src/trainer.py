import os
import torch
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence
class Trainer:
    def __init__(self,model,optimizer,args):
        self.model=model
        self.optimizer=optimizer
        self.args=args
        self.device=args.device
        self.batch_size=args.batch_size
        self.checkpoint_dir=args.checkpoint_dir
    def fit(self,train_data,valid_data,num_epochs):
        best_loss=float('inf')
        best_model=None
        for epoch in range(num_epochs):
            self.train_epoch(train_data,self.batch_size)
            val_loss=self.evaluate(valid_data)
            if val_loss<best_loss:
                # Update best model
                best_model={
                    'epoch':epoch,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                }
            if epoch%100==0:
                self.save_checkpoint(epoch)
        self.save_best_model(best_model)
            
    def train_epoch(self,train_data):
        self.model.train()
        for start_idx in range(0,len(train_data),self.batch_size):
            end_idx=min(start_idx+self.batch_size,len(train_data))
            train_data_batch = train_data[start_idx:end_idx]
            loss=self.get_loss(train_data_batch)
            # Remove old gradients
            self.optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            # Update parameters
            self.optimizer.step()
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
                train_data_batch = valid_data[start_idx:end_idx]
                loss=self.get_loss(train_data_batch)
                val_loss+=loss
                cnt+=1
            val_loss/=cnt
        return val_loss
    def save_checkpoint(self,epoch):
        checkpoint={
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()
        }
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
        checkpoint_path=os.path.join(self.checkpoint_dir,f'best_checkpoint.pth')
        torch.save(checkpoint,checkpoint_path)
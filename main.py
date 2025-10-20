import torch
from src.config import Config
from src.dataset import Dataset
from src.model import ProdLDA
from src.trainer import Trainer
def main():
    parser=Config.create_new_parser()
    Config.add_train_argument(parser)
    Config.add_data_argument(parser)
    args=parser.parse_args()
    data=Dataset(data_dir=args.data_dir,dataset=args.dataset)
    model=ProdLDA(vocab_size=len(data.vocab),hidden_size=args.hidden_size,num_topics=args.num_topics,dropout=args.dropout,use_lognormal=args.use_lognormal)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer=Trainer(model,optimizer,args)
    if args.checkpoint_path:
        trainer.load_checkpoint(args.checkpoint_path)
    else:
        trainer.fit(data.train_data,data.valid_data,args.num_epochs)

if __name__ == 'main':
    main()
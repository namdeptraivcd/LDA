import torch
from src.config import Config
from src.dataset import Dataset
from src.model import ProdLDA
from src.trainer import Trainer
def main():
    parser=Config.create_new_parser()
    Config.add_train_argument(parser)
    Config.add_eval_argument(parser)
    Config.add_data_argument(parser)
    args=parser.parse_args()
    data=Dataset(args)
    model=ProdLDA(vocab_size=len(data.vocab),hidden_size=args.hidden_size,num_topics=args.num_topics,dropout=args.dropout,use_lognormal=args.use_lognormal).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer=Trainer(model,optimizer,args,data)
    if args.checkpoint_path:
        trainer.load_checkpoint(args.checkpoint_path)
    else:
        trainer.fit(data.train_data,data.valid_data,args.num_epochs)
    trainer.test(data.train_data,data.test_data,data.vocab,args.num_top_words)

if __name__ == '__main__':
    main()
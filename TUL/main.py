from TUL import TUL
from TULDataset import TULDataset
from TulTrainer import TulTrainer
import torch
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = sys.argv[1]
model_type = sys.argv[2]
ds = TULDataset("dataset/" +dataset_name + ".csv")
Tul = TUL(vocab_size=len(ds.vocab)+1, num_users=ds.num_users,dim_inp=256, dim_out=128, num_layers=2, drop_out=0.5,model_type=model_type).to(device)
trainer = TulTrainer(
        model=Tul,
        dataset=ds,
        print_progress_every=400,
        print_accuracy_every=10,
        batch_size=8,
        learning_rate=0.00095,
        epochs=30
)
print(Tul)
trainer()
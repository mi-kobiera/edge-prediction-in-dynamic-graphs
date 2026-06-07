import torch
import time
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from models.tgn_vanilla import TGN
from utils.negative_sampling import RandomNegariveSampler
from torch_geometric.data import TemporalData

def profile_batch_size(batch_size):
    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="../datasets")
    data = dataset.get_TemporalData()
    loader = TemporalDataLoader(data, batch_size=batch_size, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = TGN(num_neighbors=10, dropout=0.1, memory_dim=50, embedding_dim=50, time_dim=50, data=data, negative_sampler=RandomNegariveSampler(data), labeling=None, device=device).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    model.on_epoch_start()
    
    t0 = time.time()
    num_events = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.train_step(batch, criterion)
        loss.backward()
        optimizer.step()
        model.after_optimizer_step()
        num_events += batch.src.size(0)
        if num_events >= 10000:
            break
    t1 = time.time()
    print(f"Batch size: {batch_size}, Time for 10000 events: {t1-t0:.4f} s")

def main():
    for bs in [100, 200, 500, 1000, 2000, 5000, 10000]:
        profile_batch_size(bs)

if __name__ == '__main__':
    main()

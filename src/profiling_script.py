import torch
import time
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from models.tgn_vanilla import TGN
from utils.negative_sampling import RandomNegariveSampler
from torch_geometric.data import TemporalData

def main():
    print("Loading data...")
    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="../datasets")
    data = dataset.get_TemporalData()
    print(f"Num edges: {data.num_events}")
    
    loader = TemporalDataLoader(data, batch_size=200, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data = data.to(device)
    sampler = RandomNegariveSampler(data)
    
    model = TGN(
        num_neighbors=10, 
        dropout=0.1, 
        memory_dim=50, 
        embedding_dim=50, 
        time_dim=50, 
        data=data, 
        negative_sampler=sampler, 
        labeling=None,
        device=device
    ).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    model.on_epoch_start()
    
    print("Starting profiling loop...")
    times = {'sample': 0, 'neighbor': 0, 'memory_gnn': 0, 'pred_loss': 0, 'update_insert': 0, 'backward': 0}
    
    for i, batch in enumerate(loader):
        if i >= 100:
            break
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        model.negative_sampler.sample(batch=batch, size=batch.src.size(0))
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        times['sample'] += t1 - t0
        
        n_id, edge_index, e_id = model.neighbor_loader(batch.n_id)
        model.assoc[n_id] = torch.arange(n_id.size(0), device=model._device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.time()
        times['neighbor'] += t2 - t1
        
        z, last_update = model.memory(n_id)
        z = model.gnn(z, last_update, edge_index, model._data.t[e_id], model._data.msg[e_id])
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t3 = time.time()
        times['memory_gnn'] += t3 - t2
        
        pos_out = model.link_pred(z[model.assoc[batch.src]], z[model.assoc[batch.dst]])
        neg_out = model.link_pred(z[model.assoc[batch.src]], z[model.assoc[batch.neg_dst]])
        loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t4 = time.time()
        times['pred_loss'] += t4 - t3
        
        loss.backward()
        optimizer.step()
        model.after_optimizer_step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t5 = time.time()
        times['backward'] += t5 - t4
        
        model.memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
        model.neighbor_loader.insert(batch.src, batch.dst)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t6 = time.time()
        times['update_insert'] += t6 - t5
        
    for k, v in times.items():
        print(f"{k}: {v:.4f} s")

if __name__ == '__main__':
    main()

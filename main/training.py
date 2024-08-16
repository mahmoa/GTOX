from torch_geometric.data import DataLoader
import warnings
import torch
import main.GCNN_model as GCNN_model
import main.preprocessing as preprocessing

warnings.filterwarnings("ignore")

model = GCNN_model.GCN()

# Binary cross-entropy loss appropriate for 0/1 classification problems
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create data loaders
data = preprocessing.get_data()
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(data, model, loader):
    '''
    train the given model for 1 epoch
    args
        - data
        - model
        - loader
    returns
        - loss
        - embedding
    '''
    # Enumerate over the data
    for batch in loader:
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch) 
        # Calculating the loss and gradients
        #print(pred, batch.y)
        loss = loss_fn(pred, (batch.y).unsqueeze(1))     
        loss.backward()  
        # Update using the gradients
        optimizer.step()
    
    return loss

# disable autogradient (no backpropagation, only forward)
@torch.no_grad()
def test(model, loader):
    pred = model(data.x, data.edge_index, data.batch)
    pred = ()

    return


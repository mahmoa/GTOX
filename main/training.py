from torch_geometric.data import DataLoader
import warnings
import torch
import main.GCNN_model as GCNN_model
import main.preprocessing as preprocessing

threshold = 0.5

warnings.filterwarnings("ignore")

def train(model, loader, mode, device):
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
    # Adam optimizer is a commmon choice for minimzing the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

    # Binary cross-entropy loss appropriate for 0/1 classification problems
    loss_fn = torch.nn.BCELoss()

    if mode == 'train':
        model.train()
    if mode =='val' or mode =='test':
        model.eval()

    # Training loop
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

#disable autogradient (no backpropagation, only forward)
@torch.no_grad()
def test(model, loader, device, results=False):
    model.eval()

    correct = 0
    total = 0
    total_preds = []

    for data in loader:
        data = data.to(device)
        x, edge_index, batch, labels = data.x, data.edge_index, data.batch, data.y
        pred = model(data.x, data.edge_index, data.batch)
        pred = (pred.squeeze() >= threshold).float()
        total_preds.append(pred)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    if results:
        accuracy = correct / total
        return accuracy
    else:
        return pred


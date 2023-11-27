import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

WHATEVER = 120 # 120
HIDDEN_DIMENSIONS = 15 # 15
HIDDEN_LAYERS = 2 # 2

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)

        self.conv1 = GCNConv(3, HIDDEN_DIMENSIONS)

        self.hidden_layers = [GCNConv(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS) for _i in range(0,HIDDEN_LAYERS)]

        self.conv3 = GCNConv(HIDDEN_DIMENSIONS, WHATEVER)        
        #self.conv4 = GCNConv(6, 2)
        #self.conv5 = GCNConv(4, 2)
        self.classifier = Linear(WHATEVER, 2)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh() #TODO: change to see what happens

        for l in self.hidden_layers:
            h = l(h, edge_index)
            h = h.tanh()

        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # h = self.conv5(h, edge_index)
        # h = h.tanh()
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h
    
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
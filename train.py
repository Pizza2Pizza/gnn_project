import torch
from dataset import MyOwnDataset
import numpy as np 
import mlflow
import mlflow.sklearn
import sys
from tqdm import tqdm
from model import GCN
from IPython.display import Javascript,display  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

act_fns = {
    "Sigmoid": torch.nn.Sigmoid(),
    "tahn": torch.nn.Tanh(),
    "ReLu": torch.nn.ReLU()
}

activation = "Sigmoid"
converter = act_fns[activation]  # needed for BCEWithLogits to get probability values
accuracy = -1

def print_output(pred, real):
    size = pred.size()

    p_zero = -1
    for i in range(0, size[0]):
        if real[i][0].item() == 1:
            p_zero = i

    probabilities = converter(pred)

    print("Predictions (actual patient 0 is node %d)" % p_zero)
    for i in range(size[0]):
        print("Node %d: prob. for: %.2f, prob. against: %.2f %s" % (
            i, 
            probabilities[i,0].item(),
            probabilities[i,1].item(), 
            "   <-- patient 0" if i == p_zero else ""
        ))

type_graph = int(sys.argv[1]) if len(sys.argv) > 1 else 1
#1 means ER graph, 2 means rgg graph
if type_graph == 1:
    train_dataset = MyOwnDataset(root="data/ER_Graph/")
    test_dataset = MyOwnDataset(root="data/ER_Graph/", test = True)
    graph_type = "ER"
elif type_graph == 2:
    train_dataset = MyOwnDataset(root="data/RGG_Graph/")
    test_dataset = MyOwnDataset(root="data/RGG_Graph/", test = True)
    graph_type = "RGG"
elif type_graph ==3:
    train_dataset = MyOwnDataset(root="data/RandomTree_Graph/")
    test_dataset = MyOwnDataset(root="data/RandomTree_Graph/", test = True)
    graph_type = "Random tree"
else:
    raise Exception("No such Graph")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Loading the model
print("Loading model...")
model = GCN()
model = model.to(device)
print(model)
print(f"Number of parameters: {count_parameters(model)}")

# Define weights
position_weight=[10,1]

position_weight1 = " "
for i in position_weight:
    position_weight1 = position_weight1 + " " +str(i) + ","
position_weight1 = position_weight1[0:-1]

# Define the optimizer and loss function
lern_rate = 0.01
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(position_weight,dtype=torch.float))  # Define loss criterion.
optimizer = torch.optim.SGD(model.parameters(), lr=lern_rate)  # Define optimizer.
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


#data = train_dataset[0]
def train_one_epoch(epoch,train_loader):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device) # Use GPU
        optimizer.zero_grad()  # Clear gradients.
        out, h = model(batch.x, batch.edge_index)  # Perform a single forward pass.
        loss = criterion(out, batch.y.float())
        #temp = data.y.unsqueeze(1).float()  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(out).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step


def my_test():
    model.eval()

      #converter = torch.nn.Sigmoid()  # needed for BCEWithLogits to get probability values
    right_prediction = 0
      # Check against ground-truth labels.
    for idx in range(len(test_dataset)):
        
        data = test_dataset[idx]
        out,h = model(data.x, data.edge_index)
        #print_output(out, data.y)
        temp = []
        probabilities = converter(out)
        for i in range(out.size()[0]):
            temp.append(probabilities[i,0].item() - probabilities[i,1].item())
        p_0 = np.argmax(temp) #gives the index of the node with highest probability
        actual_p_0 = np.argmax(data.y, axis = 0)[0]
        if p_0 == actual_p_0.item():
            right_prediction += 1
    accuracy = right_prediction/len(test_dataset)

    return accuracy

def test(epoch,test_loader):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)  
        pred, h = model(batch.x,batch.edge_index) 
        loss = criterion(pred, batch.y.float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    
    return running_loss/step

size = len(train_dataset)
epochs = 100
# for epoch in range(epochs):    
#     for idx in range(size):    
#         data = train_dataset[idx]
        

#         loss, h = train(data)
#         #if epoch % 50 == 0 and idx == size-1:
#             #visualize_embedding(h, color=[e[0] for e in data.y], epoch=epoch, loss=loss)
#             #time.sleep(0.3)

def calculate_metrics(y_pred, y_true, epoch, type):
    global accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")


def run_one_training(train_dataset, test_dataset):
    # Loading the dataset
    print("Loading dataset...")


    # Prepare training
    BATCH_SIZE = 20
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Start training
    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(epochs): 
        if early_stopping_counter <= 10: # = x * 5 
            # Training
            model.train()
            loss = train_one_epoch(epoch,train_loader)
            print(f"Epoch {epoch} | Train Loss {loss}")
            

            # Testing
            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch,test_loader)
                print(f"Epoch {epoch} | Test Loss {loss}")
               
                
                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]


num_nodes = test_dataset[0].num_nodes
days = test_dataset[0].days

with mlflow.start_run():
    mlflow.set_tag("model_name", "gnn")

    params = {
        "days": days,
        "num_nodes": num_nodes,
        "learning_rate": lern_rate,
        "weight_1": position_weight[0],
        "weight_2": position_weight[1],
        "graph_type": graph_type,
        "activ_funcntion": activation
    }

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
   # mlflow.log_param("days", days)
    # mlflow.log_param("num_nodes", num_nodes)
   # mlflow.log_metric("weight_1", position_weight[0])
   # mlflow.log_metric("weight_2", position_weight[1])
   # mlflow.log_metric("learning rate", lern_rate)
    #mlflow.log_metric("graph_type", graph_type)

    mlflow.pytorch.log_model(model,"gnn")
    print("test1")
print("test1")


run_one_training(train_dataset, test_dataset)
test_acc = my_test()
print(f'My Test Accuracy: {test_acc:.4f}')




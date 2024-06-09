
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn import Dropout, Module, CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import NeighborLoader
from torch.nn.functional import relu, log_softmax
from typing import Tuple, List, Dict
from torch import manual_seed, cuda
from torch.optim import Adam
import torch_geometric
import numpy as np
import random
import torch

    
class GCN(Module):
    """
    GCN-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    def __init__(self, in_features: int, hidden_size: int, output_dimension: int, dropout: float, seed: int) -> None:
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        Args:
        - in_features (int): Number of input features
        - hidden_size (int): Number of hidden units
        - output_dimension (int): Number of output classes
        - dropout (float): Dropout probability
        - seed (int): Random seed
        """
        #Here we initialize the model object
        super(GCN, self).__init__()
        #Here we set the seed
        manual_seed(seed)
        #Here we create the first convolutional layer
        self.conv1 = GCNConv(in_features, hidden_size)
        #Here we create the second convolutional layer
        self.conv2 = GCNConv(hidden_size, output_dimension)
        #Here we create the dropout layer
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Define forward step of network. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution. Finally apply softmax to get class logits.

        Args:
        - x (torch.Tensor): Input tensor
        - edge_index (torch.Tensor): Graph edge index tensor

        Returns:
        - torch.Tensor: Output tensor
        """
        #Here we pass the input through the first convolutional layer
        x = self.conv1(x, edge_index) 
        #Here we apply the relu activation function
        x = relu(x) 
        #Here we apply the dropout layer
        x = self.dropout(x) 
        #Here we pass the input through the second convolutional layer
        x = self.conv2(x, edge_index) 
        #Here we apply the softmax activation function to get each category's probability
        return log_softmax(x, dim=1) 
    
class GraphSAGE(Module):
    """
    GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    def __init__(self, in_features: int, hidden_size: int, output_dimension: int, dropout: float, seed: int) -> None:
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        Args:
        - in_features (int): Number of input features
        - hidden_size (int): Number of hidden units
        - output_dimension (int): Number of output classes
        - dropout (float): Dropout probability
        - seed (int): Random seed
        """
        #Here we initialize the model object
        super(GraphSAGE, self).__init__()
        
        #Here we set the seed
        manual_seed(seed)
        #Here we create the first convolutional layer
        self.sage1 = SAGEConv(in_features, hidden_size*2, aggr='mean')
        #Here we create the second convolutional layer
        self.sage2 = SAGEConv(hidden_size*2, hidden_size, aggr='mean')
        #Here we create the first dropout layer
        self.dropout = Dropout(p=dropout)
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Define forward step of network. In this example, pass inputs through convolution, 
        apply relu and dropout, then pass through second convolution. Finally apply softmax to get class logits.
        
        Args:
        - x (torch.Tensor): Input tensor
        - edge_index (torch.Tensor): Graph edge index tensor

        Returns:
        - torch.Tensor: Output tensor
        """
        #Here we pass the input through the first convolutional layer
        x = self.sage1(x, edge_index)
        #Here we apply the relu activation function
        x = relu(x)
        #Here we apply the dropout layer
        x = self.dropout(x)
        #Here we pass the input through the second convolutional layer
        x = self.sage2(x, edge_index)
        #Here we apply the softmax activation function to get each category's probability
        return log_softmax(x, dim=1) 
    
    
class GAT(Module):
    """
    GAT-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    HIDDEN_ATTENTION_HEADS = 8
    OUTPUT_ATTENTION_HEADS = 1
    def __init__(self, in_features: int, hidden_size: int, output_dimension: int, dropout: float, seed: int) -> None:
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        Args:
        - in_features (int): Number of input features
        - hidden_size (int): Number of hidden units
        - output_dimension (int): Number of output classes
        - dropout (float): Dropout probability
        - seed (int): Random seed
        """
        #Here we initialize the model object
        super(GAT, self).__init__()
        #Here we set the seed
        manual_seed(seed)
        #Here we create the first convolutional layer
        self.gat1 = GATv2Conv(in_features, hidden_size, heads=self.HIDDEN_ATTENTION_HEADS)
        #Here we create the second convolutional layer
        self.gat2 = GATv2Conv(hidden_size*self.HIDDEN_ATTENTION_HEADS, output_dimension, heads=self.OUTPUT_ATTENTION_HEADS)
        #Here we create the dropout layer
        self.dropout = Dropout(dropout)
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Define forward step of network. In this example, pass inputs through convolution, 
        apply relu and dropout, then pass through second convolution. Finally apply softmax to get class logits.

        Args:
        - x (torch.Tensor): Input tensor
        - edge_index (torch.Tensor): Graph edge index tensor

        Returns:
        - torch.Tensor: Output tensor
        """
        #Here we pass the input through the first convolutional layer
        x = self.gat1(x, edge_index)
        #Here we apply the relu activation function
        x = relu(x)
        #Here we apply the dropout layer
        x = self.dropout(x)
        #Here we pass the input through the second convolutional layer
        x = self.gat2(x, edge_index)
        #Here we apply the softmax activation function to get each category's probability
        return log_softmax(x, dim=1) 
    

class NodeClassification():
    NUM_NEIGHBORS = [5, 10]
    BATCH_SIZE = 16
    def __init__(self, data: torch_geometric.data.Data, hyperparameters: dict) -> None:
        """
        Initialize the NodeClassification object. This object is used to train and evaluate a GNN model.

        Args:
        - data (torch_geometric.data.Data): Data object
        - model (str): Model type
        - hyperparameters (dict): Hyperparameters
        """
        self.data = data
        self.hyperparameters = hyperparameters
        self.net, self.optimizer = self.__initialize_model()
        self.criterion = CrossEntropyLoss()
    
    def training_neighbor_sampling(self, verbose: bool=True) -> Tuple[List[float], List[float]]:
        """
        Function to train the GNN model with neighbor sampling.

        Args:
        - verbose (bool): Whether to print training updates

        Returns:
        - list: Training losses
        - list: Validation losses
        """
        #Here we get the seed
        seed = self.hyperparameters["seed"]
        #Here we print the seed
        print(f'Setting seed to {seed}...')
        #Here we set the seed
        self.__set_seed(seed)

        #Here we get the number of epochs
        number_epochs = self.hyperparameters['number_epochs']

        #Here we define the train loader
        #The number of neighbors are the number of neighbors to sample from the graph
        #The batch size is the number of samples to use in each iteration
        #The input nodes are the nodes to sample from
        train_loader = NeighborLoader(
            self.data,
            num_neighbors=self.NUM_NEIGHBORS,
            batch_size=self.BATCH_SIZE,
            input_nodes=self.data.train_mask)

        #Here we initialize lists to store the training and validation losses
        train_losses = []
        val_losses = []

        # This is the training loop
        for epoch in range(number_epochs):
            #Here we define the loss and validation loss of each epoch
            total_loss = 0
            total_val_loss = 0
            #Here we iterate over the train loader
            for batch in train_loader:
                #Here we set the model to train mode
                self.net.train()
                #Here we zero the gradients
                self.optimizer.zero_grad()
                #Here we perform a forward pass
                probs = self.net(batch.x, batch.edge_index)
                #Here we compute the loss solely based on the training nodes.
                loss = self.criterion(probs[batch.train_mask], batch.y[batch.train_mask])
                #Here we backpropagate the gradients
                loss.backward()
                #Here we update the weights
                self.optimizer.step()
                #Here we compute the loss based on the validation nodes
                val_loss = self.criterion(probs[batch.val_mask], batch.y[batch.val_mask])
                #Here we save the training and validation losses
                total_loss += loss
                total_val_loss += val_loss

            #Here we save the training and validation losses
            train_losses.append(total_loss.item()/ len(train_loader))
            val_losses.append(total_val_loss.item()/ len(train_loader))

            #Here we check if we should print the results
            if (verbose == True) & ( epoch % 50 == 0):
                print(f'Epoch: {epoch} | Loss: {loss/ len(train_loader)}')

        return train_losses, val_losses

    
    def training(self, verbose: bool=True) -> Tuple[List[float], List[float]]:
        """
        Function to train the GNN model.

        Args:
        - verbose (bool): Whether to print training updates

        Returns:
        - list: Training losses
        - list: Validation losses
        """
        
        #Here we get the seed
        seed = self.hyperparameters["seed"]
        #Here we print the seed
        print(f'Setting seed to {seed}...')
        #Here we set the seed
        self.__set_seed(seed)

        #Here we get the number of epochs
        number_epochs = self.hyperparameters['number_epochs']

        #Here we initialize lists to store the training and validation losses
        train_losses = []
        val_losses = []

        # This is the training loop
        for epoch in range(number_epochs):
             #Here we set the model to train mode
            self.net.train()
             #Here we zero the gradients
            self.optimizer.zero_grad()
            
            #Here we perform a single forward pass
            probs = self.net(self.data.x, self.data.edge_index)
             #Here we compute the loss solely based on the training nodes.
            loss = self.criterion(probs[self.data.train_mask], self.data.y[self.data.train_mask])
            #Here we backpropagate the gradients
            loss.backward()
            #Here we update the weights
            self.optimizer.step()
            #Here we compute the loss based on the validation nodes
            val_loss = self.criterion(probs[self.data.val_mask], self.data.y[self.data.val_mask])
            #Here we save the training and validation losses
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

             #Here we check if we should print the results
            if (verbose == True) & ( epoch % 50 == 0):
                print(f'Epoch: {epoch} | Loss: {loss.item()}')
        
        # Here we return the probabilities, the best coloring, the best soft loss and the best hard loss
        return train_losses, val_losses
    
    def test(self) -> Dict[str, float]:
        """
        Function to test the GNN model.

        Returns:
        - dict: Evaluation metrics
        """
        #Here we set the model to evaluation mode
        self.net.eval()
        #Here we perform a forward pass
        out = self.net(self.data.x, self.data.edge_index)
        #Here we get the predictions
        pred = out.argmax(dim=1)
        #Here we return the evaluation metrics
        return self.__evaluate_metrics(pred)
    
    def __initialize_model(self) -> Tuple[Module, Adam]:
        """
        Helper function to load in the GNN model and ADAM optimizer and initial embedding.
        """
        #Here we get the seed
        seed = self.hyperparameters["seed"]
        #Here we print the seed
        print(f'Setting seed to {seed}...')
        #Here we set the seed
        self.__set_seed(seed)

        #Here we get the model
        model = self.hyperparameters['model']
        #Here we get the embedding dimension
        initial_dim = self.hyperparameters['initial_dim']
        #Here we get the hidden dimension
        hidden_dim = self.hyperparameters['hidden_dim']
        #Here we get the dropout probability
        dropout = self.hyperparameters['dropout']
        #Here we get the output dimension
        out_dim = self.hyperparameters['out_dim']
        #Here we get the learning rate
        learning_rate = self.hyperparameters['learning_rate']
        #Here we get the weight decay
        weight_decay = self.hyperparameters['weight_decay']
        #Here we get the torch device
        torch_device = self.hyperparameters['device']
        #Here we get the torch dtype
        torch_dtype = self.hyperparameters['dtype']
    

        # Here we initialize the model
        if model == "GCN":
            print(f'Building {model} model...')
            #Here we create the GCN model
            net = GCN(initial_dim, hidden_dim, out_dim, dropout, seed)
        elif model == "GraphSAGE":
            print(f'Building {model} model...')
            #Here we create the GraphSAGE model
            net = GraphSAGE(initial_dim, hidden_dim, out_dim, dropout, seed)

        elif model == "GAT":
            print(f'Building {model} model...')
            #Here we create the GAT model
            net = GAT(initial_dim, hidden_dim, out_dim, dropout, seed)
        else:
            #Here we raise an error if the model type is invalid
            raise ValueError("Model only supports 'GCN', 'GraphSAGE' or 'GAT'")

        #Here we set the model to the device and dtype
        net = net.type(torch_dtype).to(torch_device)
        
        # Here we initialize the optimizer
        print('Building ADAM optimizer...')
        optimizer = Adam(net.parameters(), lr= learning_rate, weight_decay= weight_decay)

        #Here we return the model and the optimizer
        return net, optimizer
    

    def __evaluate_metrics(self, pred: torch.Tensor) -> Dict[str, float]:
        """
        Function to evaluate the model using accuracy, precision, recall, and F1 score.

        Args:
        - pred (torch.Tensor): Predictions

        Returns:
        - dict: Evaluation metrics
        """
        # Ensure the mask and predictions are on the same device
        test_mask = self.data.test_mask.to(pred.device)
        y_test = self.data.y.to(pred.device)

        # Extract the test set predictions and true labels
        pred_test = pred[test_mask]
        y_test = y_test[test_mask]

        # Calculate correct predictions for accuracy
        correct = (pred_test == y_test)
        correct_count = correct.sum().item()
        total_count = test_mask.sum().item()

        # Calculate accuracy
        accuracy = correct_count / total_count

        # Convert tensors to numpy arrays for sklearn metrics
        pred_test_np = pred_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Calculate precision, recall, and F1 score using sklearn
        precision = precision_score(y_test_np, pred_test_np, average='macro')
        recall = recall_score(y_test_np, pred_test_np, average='macro')
        f1 = f1_score(y_test_np, pred_test_np, average='macro')

        # Return all metrics in a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def __set_seed(self, seed: int) -> None:
        """
        Sets random seeds for training.
        This is the ONLY function that I didn't write myself. 
        It was taken from the Schuetz et al. repository even though it is not fundamental to the functioning of the models.

        Args: 
        - seed (int): Random seed
        """
        #Here we set the seed for the random module
        random.seed(seed)
        #Here we set the seed for the numpy module
        np.random.seed(seed)
        #Here we set the seed for the torch module
        manual_seed(seed)
        if cuda.is_available():
            #Here we set the seed for the cuda module
            cuda.manual_seed_all(seed)

class LinkPredGCN(Module):
    """
    GCN-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters.
    """
    def __init__(self, in_features: int, hidden_size: int, output_dimension: int, dropout: float, seed: int) -> None:
        """
        Initialize the model object. Establishes model architecture and relevant hyperparameters.

        Args:
        - in_features (int): Number of input features
        - hidden_size (int): Number of hidden units
        - output_dimension (int): Number of output classes
        - dropout (float): Dropout probability
        - seed (int): Random seed
        """
        #Here we initialize the model object
        super(LinkPredGCN, self).__init__()
        #Here we set the seed
        manual_seed(seed)
        #Here we create the first convolutional layer
        self.conv1 = GCNConv(in_features, hidden_size)
        #Here we create the second convolutional layer
        self.conv2 = GCNConv(hidden_size, output_dimension)
        #Here we create the dropout layer
        self.dropout = Dropout(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Define the encoder step of the network. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        Args:
        - x (torch.Tensor): Input tensor
        - edge_index (torch.Tensor): Graph edge index tensor

        Returns:
        - torch.Tensor: Encoded tensor
        """
        #Here we pass the input through the first convolutional layer
        x = self.conv1(x, edge_index) 
        #Here we apply the relu activation function
        x = relu(x) 
        #Here we apply the dropout layer
        x = self.dropout(x) 
        #Here we pass the input through the second convolutional layer
        x = self.conv2(x, edge_index) 
        #Here we apply the softmax activation function to get each category's probability
        return x
    
    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Define the decoder step of the network. In this example, we take the embeddings and edge indexes to index the edges.
        Then we obtain the dot product between source and target nodes of the edges.

        Args:
        - z (torch.Tensor): Encoded tensor
        - edge_label_index (torch.Tensor): Edge index tensor

        Returns:
        - torch.Tensor: Decoded tensor

        """
        #Here we get the source and target nodes of the edges
        src, dst = edge_label_index
        #Here we obtain the dot product between source and target nodes of the edges
        return (z[src] * z[dst]).sum(dim=1)


class LinkPrediction():
    def __init__(self, train_data: torch_geometric.data.Data, hyperparameters: dict) -> None:
        """
        Initialize the LinkPrediction object. This object is used to train and evaluate a GNN model for link prediction.

        Args:
        - train_data (torch_geometric.data.Data): Training data object
        - hyperparameters (dict): Hyperparameters
        """
        self.train_data = train_data
        self.hyperparameters = hyperparameters
        self.net, self.optimizer = self.__initialize_model()
        self.criterion = BCEWithLogitsLoss()

    def __initialize_model(self) -> Tuple[Module, Adam]:
        """
        Helper function to load in the GNN model and ADAM optimizer and initial embedding.
        """
        #Here we get the seed
        seed = self.hyperparameters["seed"]
        #Here we print the seed
        print(f'Setting seed to {seed}...')
        #Here we set the seed
        self.__set_seed(seed)

        #Here we get the model
        model = self.hyperparameters['model']
        #Here we get the embedding dimension
        initial_dim = self.hyperparameters['initial_dim']
        #Here we get the hidden dimension
        hidden_dim = self.hyperparameters['hidden_dim']
        #Here we get the dropout probability
        dropout = self.hyperparameters['dropout']
        #Here we get the output dimension
        out_dim = self.hyperparameters['out_dim']
        #Here we get the learning rate
        learning_rate = self.hyperparameters['learning_rate']
        #Here we get the weight decay
        weight_decay = self.hyperparameters['weight_decay']
        #Here we get the torch device
        torch_device = self.hyperparameters['device']
        #Here we get the torch dtype
        torch_dtype = self.hyperparameters['dtype']
    

        # Here we initialize the model
        if model == "GCN":
            print(f'Building {model} model...')
            #Here we create the GCN model
            net = LinkPredGCN(initial_dim, hidden_dim, out_dim, dropout, seed)
        else:
            #Here we raise an error if the model type is invalid
            raise ValueError("Model only supports 'GCN'")

        #Here we set the model to the device and dtype
        net = net.type(torch_dtype).to(torch_device)
        
        # Here we initialize the optimizer
        print('Building ADAM optimizer...')
        optimizer = Adam(net.parameters(), lr= learning_rate, weight_decay= weight_decay)

        #Here we return the model and the optimizer
        return net, optimizer
    
    def __set_seed(self, seed: int) -> None:
        """
        Sets random seeds for training.
        This is the ONLY function that I didn't write myself. 
        It was taken from the Schuetz et al. repository even though it is not fundamental to the functioning of the models.

        Args: 
        - seed (int): Random seed
        """
        #Here we set the seed for the random module
        random.seed(seed)
        #Here we set the seed for the numpy module
        np.random.seed(seed)
        #Here we set the seed for the torch module
        manual_seed(seed)
        if cuda.is_available():
            #Here we set the seed for the cuda module
            cuda.manual_seed_all(seed)
    
    def training(self, verbose: bool=True) -> List[float]:
        """
        Function to train the GNN model.

        Args:
        - verbose (bool): Whether to print training updates

        Returns:
        - list: Training losses
        """
        
        #Here we get the seed
        seed = self.hyperparameters["seed"]
        #Here we print the seed
        print(f'Setting seed to {seed}...')
        #Here we set the seed
        self.__set_seed(seed)

        #Here we get the number of epochs
        number_epochs = self.hyperparameters['number_epochs']

        #Here we initialize lists to store the training loss
        train_losses = []

        #This is the training loop
        for epoch in range(number_epochs):
            #Here we set the model to train mode
            self.net.train()
            #Here we zero the gradients
            self.optimizer.zero_grad()
            #Here we perform the encoding step
            z = self.net.encode(self.train_data.x, self.train_data.edge_index)
            #Here we add negative edges by using the negative sampling function
            #First we get the negative edges indexes
            negative_edges_index = negative_sampling(self.train_data.edge_index, 
                                                     num_nodes=self.train_data.num_nodes,
                                                        num_neg_samples=self.train_data.edge_index.size(1))
            #Now we put together the positive and negative edges
            edge_label_index = torch.cat([self.train_data.edge_label_index, negative_edges_index], dim=-1)
            #Now we create labels for positive and negative edges (1 for positive, 0 for negative).
            #This is done to perform the binary classification task
            #We do this by creating a tensor of ones for the positive edges and zeros for the negative edges
            edge_labels = torch.cat([self.train_data.edge_label, torch.zeros(negative_edges_index.size(1), dtype=torch.long)], dim=0)
            #Now we perform the decoding step
            #The view function is used to reshape the tensor in order to match the shape of the labels
            decoded = self.net.decode(z, edge_label_index).view(-1)
            #Here we compute the loss
            loss = self.criterion(decoded, edge_labels)
            #Here we backpropagate the gradients
            loss.backward()
            #Here we update the weights
            self.optimizer.step()
            #Here we save the training loss
            train_losses.append(loss.item())
            
            #Here we check if we should print the results
            if (verbose == True) & ( epoch % 50 == 0):
                print(f'Epoch: {epoch} | Loss: {loss.item()}')

        return train_losses
    
    def test(self, test_data: torch_geometric.data.Data) -> Dict[str, float]:
        """
        Function to test the GNN model.

        Args:
        - test_data (torch_geometric.data.Data): Test data object

        Returns:
        - dict: Evaluation metrics
        """
        #Here we set the model to evaluation mode
        self.net.eval()
        #Here we perform the encoding step
        z = self.net.encode(test_data.x, test_data.edge_index)
        #Here we perform the decoding step
        decoded = self.net.decode(z, test_data.edge_label_index).view(-1)
        #Here we get the predictions using a sigmoid function
        pred = torch.sigmoid(decoded)

        #Here we return the evaluation metrics
        return self.__evaluate_metrics(pred, test_data.edge_label)
    
    def __evaluate_metrics(self, pred: torch.Tensor, edge_labels: torch.Tensor) -> Dict[str, float]:
        """
        Function to evaluate the model using ROC AUC, MRR, and Hits@K.

        Args:
        - pred (torch.Tensor): Predictions
        - edge_labels (torch.Tensor): Edge labels

        Returns:
        - dict: Evaluation metrics
        """
        #Here we get the predictions
        pred = pred.cpu().detach().numpy()
        #Here we get the edge labels
        edge_labels = edge_labels.cpu().detach().numpy()
        #Here we calculate the ROC AUC
        roc_auc = roc_auc_score(edge_labels, pred)
        #Here we calculate the Hits@10 and MRR
        hits_at_1 = self.__hitsatk(edge_labels, pred, 2)
        #Here we return the evaluation metrics
        return {
            'roc_auc': roc_auc,
            'hits_at_1': hits_at_1
        }
    
    def __hitsatk(self, edge_labels: np.ndarray, pred: np.ndarray, k: int) -> float:
        """
        Helper function to calculate Hits@K.

        Args:
        - edge_labels (np.ndarray): Edge labels
        - pred (np.ndarray): Predictions
        - k (int): Number of hits

        Returns:
        - float: Hits@K
        """
        #Here we get the indexes of the sorted predictions
        indexes = np.argsort(pred)[::-1]
        #Here we get the top k indexes
        top_k_indexes = indexes[:k]
        #Here we get the top k labels
        top_k_labels = edge_labels[top_k_indexes]
        #Here we get the number of hits
        hits = np.sum(top_k_labels)
        #Here we return the hits@k
        return hits / k
        

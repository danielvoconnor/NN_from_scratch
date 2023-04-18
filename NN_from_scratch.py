import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import sklearn.datasets

def softmax(u):
    term = np.exp(u - np.max(u, axis=0))
    return term/term.sum(axis=0)

def log_sum_exp(u):
    M = np.max(u, axis=0)
    return M + np.log(np.sum(np.exp(u - M), axis=0))

def cross_entropy_from_logits(u, y):
    return log_sum_exp(u) - np.sum(y * u, axis=0)

def relu(u, prime=False): 
    if prime:
        return np.where(u > 0, 1., 0.)
    else: 
        return np.maximum(u,0)

def forward(x, W, activs):
    layer_outputs = [x] 
    for i, a in enumerate(activs):
        Wi = W[i]
        layer_outputs.append(Wi[:,1:] @ a(layer_outputs[i]) + Wi[:,0][:,None])
           
    return layer_outputs

def compute_grad(W, activs, x, y, grad):
    layer_outputs = forward(x, W, activs)
    
    batch_size = x.shape[1]
    term = softmax(layer_outputs[-1]) - y
    other_term = activs[-1](layer_outputs[-2]) 
    grad[-1][:,0] = np.sum(term, axis=1) / batch_size
    grad[-1][:,1:] = term @ other_term.T / batch_size
    
    for i in reversed(range(1, len(W))):
        term = activs[i](layer_outputs[i], prime=True) * (W[i][:,1:].T @ term)
        other_term = activs[i-1](layer_outputs[i-1])
        grad[i-1][:,0] = np.sum(term, axis=1) / batch_size
        grad[i-1][:,1:] = term @ other_term.T / batch_size

def avg_cross_entropy(W, activs, X, y):
    layer_outputs = forward(X, W, activs)
    cross_entropy_vals = cross_entropy_from_logits(layer_outputs[-1], y)
    
    return np.mean(cross_entropy_vals)

def initialize_weights_Kaiming(nodes_per_layer, input_dim):       
    W = []
    for output_dim in nodes_per_layer:
        Wi = np.random.randn(output_dim, input_dim+1) * np.sqrt(2/input_dim)
        Wi[:,0] = 0.0
        W.append(Wi)
        input_dim = output_dim 
    
    return W
                
def train_model(params, X_train, y_train, X_val, y_val): 
    nodes_per_layer = params['nodes_per_layer']
    activs = [lambda u:u] + params['activations']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    
    num_features, num_examples = X_train.shape  
    W = initialize_weights_Kaiming(nodes_per_layer, num_features)
    grad = [np.zeros(w.shape) for w in W]

    ace_history = {'train':[], 'val':[]} # ace = "average cross-entropy"
    
    for ep in range(num_epochs):        
        shuffled_idxs = np.random.permutation(num_examples)
        for start_idx in range(0, num_examples, batch_size):
            stop_idx = np.minimum(start_idx + batch_size, num_examples)
            x_batch = X_train[:, start_idx:stop_idx]
            y_batch = y_train[:, start_idx:stop_idx]
            
            compute_grad(W, activs, x_batch, y_batch, grad)
            
            for j in range(len(W)):   
                W[j] -= learning_rate*grad[j]
                
        ace_train = avg_cross_entropy(W, activs, X_train, y_train)
        ace_val = avg_cross_entropy(W, activs, X_val, y_val)
        ace_history['train'].append(ace_train)
        ace_history['val'].append(ace_val)
        
        print(f'Just finished epoch {ep}')
        print(f'Avg. cross-entropy (training data): {ace_train:.6}')
        print(f'Avg. cross-entropy (validation data): {ace_val:.6}') 
        
    return W, ace_history

def predict(W, activs, X):
    activs.insert(0, lambda u:u)
    u = forward(X, W, activs)
    probs = softmax(u[-1])
    labels_pred = np.argmax(probs, axis=0)
        
    return labels_pred

def prepare_MNIST():
    dataset = sk.datasets.fetch_openml('mnist_784') # Downloading the data is slow.
    X, labels = np.float64(dataset.data)/255, np.int64(dataset.target)
    
    y = np.zeros((len(X), 10))
    y[np.arange(len(y)), labels] = 1.0
    
    N_train = 60000
    X_train, y_train, labels_train = X[0:N_train], y[0:N_train], labels[0:N_train]
    X_val, y_val, labels_val = X[N_train:], y[N_train:], labels[N_train:]
        
    return X_train.T, y_train.T, labels_train, X_val.T, y_val.T, labels_val
    
if __name__ == '__main__':
    print('Loading MNIST data...')
    X_train, y_train, labels_train, X_val, y_val, labels_val = prepare_MNIST()
    print('Finished loading MNIST data. Training begins now.')
    
    # Specify neural network architecture and hyperparameter values
    nodes_per_layer = [128, 64, 32, 10]
    num_layers = len(nodes_per_layer)
    activs = [relu]*(num_layers - 1) # Activations for the hidden layers
    
    params = {'nodes_per_layer':nodes_per_layer, 
              'activations':activs,
              'num_epochs':15, 
              'batch_size':100,
              'learning_rate':.05}
    
    # Train the model and check accuracy on validation dataset
    W, ace_history = train_model(params, X_train, y_train, X_val, y_val)
       
    plt.figure()
    plt.plot(ace_history['train'], label='Training data')
    plt.plot(ace_history['val'], label='Validation data')
    plt.legend()
    plt.title('Avg. cross-entropy vs. epoch')
    plt.show()
    
    labels_pred = predict(W, activs, X_val)
    acc = np.mean(labels_pred == labels_val)
    print(f'Classification accuracy (validation data): {acc:.6}')










    

    
        
        

            
                
            
                
    
        
    
    
    
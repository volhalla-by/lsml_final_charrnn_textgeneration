import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def one_hot_encode(arr, n_labels):

    one_hot = np.eye(n_labels)[arr].astype(np.float32)

    return one_hot


def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)


    inputs = inputs.to(DEVICE)

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    p = p.cpu() # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):

    net.to(DEVICE)

    net.eval() # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=512, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Define the LSTM layer
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        # Get the outputs and the new hidden state from the lstm
        out, hidden = self.lstm(x, hidden)

        # Pass through a dropout layer
        out = self.dropout(out)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        # Put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(DEVICE),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(DEVICE))

        return hidden


def load_mlflow_model(MLFLOW_SERVER_URL, experiment_name):
    # last run of the experiment from the list of all runs
    mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
    client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URL)
    experiment = client.get_experiment_by_name(experiment_name)      
    run_info = client.list_run_infos(experiment.experiment_id)[-1]

    #load model
    model = mlflow.pytorch.load_model("runs:/{}/model".format(run_info.run_id))

    return model


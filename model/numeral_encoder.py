"""
This file includes information about the numeral encoder. The numeral encoder is a neural network that encodes a number into a vector of numbers.
Here we are going to implement several different types of numeral encoders. These include:
    
    - a positional encoder that is a simple positional encoder that encodes the number into a vector of numbers. Adopted from the paper "Attention is all you need".
    - a number encoder called Deterministic, Independent-of-Corpus Embeddings (DICE)
    - a GMM model that first use EM methods to learn the representations and then encode the numbers as the weights of the GMM model.

    - a recurrent neural network called digit-RNN. 

"""

import torch
import numpy as np
import networkx as nx
from torch import nn
from torch.nn import CrossEntropyLoss
from sklearn.mixture import GaussianMixture
import pickle



class QuantileScaling():
    def __init__(self, train_values):
        self.train_values_sorted = np.sort(train_values, axis=None)
        self.train_values_sorted = torch.from_numpy(self.train_values_sorted).float()
       
    
    def quantile_scaling_function(self, x):

        self.train_values_sorted = self.train_values_sorted.to(x.device)

        # print("x: ", x)
        # print("train_values_sorted: ", self.train_values_sorted)


        num_smaller = torch.sum((x.reshape(-1, 1) - self.train_values_sorted.reshape(1, -1) > 0), dim=1)

        # print("num_smaller: ", num_smaller)
       
    

        largest_smaller_index = num_smaller - 1 
        # print("largest_smaller_index: ", largest_smaller_index)
        

        largest_smaller_index[largest_smaller_index < 0] = 0
        # print("value shape: ", self.train_values_sorted.shape[0])
        # largest_smaller_index[largest_smaller_index > 0] = 0
        largest_smaller_index[ largest_smaller_index >= self.train_values_sorted.shape[0] - 2] = self.train_values_sorted.shape[0] - 2
       
        next_index = largest_smaller_index + 1

        largest_smaller_index_value = self.train_values_sorted[largest_smaller_index]
        next_index_value = self.train_values_sorted[next_index]

        approximate_indices = largest_smaller_index + (x - largest_smaller_index_value)  / (next_index_value - largest_smaller_index_value)

        percentile = approximate_indices / len(self.train_values_sorted)

        percentile[percentile < 0] = 0
        percentile[percentile > 1] = 1

    
        return percentile * 64

def log_scaling_function(x):
    # print("x: " + str(x))
    x[ x > 1] = torch.log(x[ x > 1]) + 1
    x[ x < -1] = - torch.log(-x[ x < -1]) - 1
    # print("x: " + str(x))
    return x

def log_scaling_function_np(x):
    # print("x: " + str(x))
    x[ x > 1] = np.log(x[ x > 1]) + 1
    x[ x < -1] = - np.log(-x[ x < -1]) - 1
    # print("x: " + str(x))
    return x

class NumeralEncoder(torch.nn.Module):
    """
    This class is the base class for all numeral encoders.
    """
    def __init__(self, output_size, train_values, all_values):
        super(NumeralEncoder, self).__init__()

        self.output_size = output_size

        self.train_values = train_values
        self.all_values = all_values
            
        value_vocab = dict(zip(all_values, range(0, len(all_values))))
        self.value_vocab = value_vocab

    
    def get_embeddings(self):
        """
        This function returns the embeddings of the test set.
        """
        # if torch.cuda.is_available() :
        #     return self.forward( torch.tensor(self.all_values).cuda() )
        # else:
        #     return self.forward( torch.tensor(self.all_values) )
        

        return self.forward( self.all_values)


    def values2ids(self, values: list):
        """
        This function converts the values of the numeral encoder to ids.
        """
        ids = []

        for value in values:
            # print("values: " + str(value))
            # print("value2id: " + str(self.value_vocab))
            ids.append(self.value_vocab[value])
        return ids
    
    def value2id(self, value):
        """
        This function converts a value to an id.
        """
        return self.value_vocab[value]
    

    def forward(self, x):
        """
        This function is the forward pass of the numeral encoder.
        """
        raise NotImplementedError

    


class PositionalEncoder(NumeralEncoder):
    """
    This class is the positional encoder.
    """
    def __init__(self, output_size, train_values, all_values, n=10000, scaler="log"):
        super(PositionalEncoder, self).__init__(output_size, train_values, all_values)
        self.n = n

        self.scaler_name  = scaler

        if scaler == "log":
            self.scaler = log_scaling_function
        elif scaler == "quantile":
            self.scaler = QuantileScaling(self.train_values).quantile_scaling_function
        

        if torch.cuda.is_available() :
            self.sinosoidal_embedding = torch.nn.Embedding(len(self.all_values), output_size).cuda()
        else:
            self.sinosoidal_embedding = torch.nn.Embedding(len(self.all_values), output_size)

       
        
        all_values_tensor =torch.tensor(self.all_values)
        if torch.cuda.is_available() :
            all_values_tensor = all_values_tensor.cuda()

        
        d = self.output_size


        x = self.scaler(all_values_tensor)
        denominator = 1 / torch.pow(self.n, 2 * torch.arange(0, d // 2).float().to(x.device) / d).unsqueeze(0)
        x = x.unsqueeze(1)

        # print("x: ", x.shape)
        # print("denominator: ", denominator.shape)
        # print(self.sinosoidal_embedding.weight.shape)
        self.sinosoidal_embedding.weight.requires_grad = False
        self.sinosoidal_embedding.weight[:, 0::2] = torch.sin(x * denominator)
        self.sinosoidal_embedding.weight[:, 1::2] = torch.cos(x * denominator)

        

        
    def forward(self, x):
        """
        This function is the forward pass of the positional encoder.
        
        x: the input array of numbers to be encoded
        """

        if isinstance(x, list):
            x_ids = self.values2ids(x)
        else:
            x_ids =  self.values2ids([x])

        if torch.cuda.is_available() :
            x_ids = torch.tensor(x_ids).cuda()
        else:
            x_ids = torch.tensor(x_ids)
        
        return self.sinosoidal_embedding(x_ids)

    



    # def forward(self, x):
    #     """
    #     This function is the forward pass of the positional encoder.
        
    #     x: the input array of numbers to be encoded
    #     """
    
       

    #     x = self.scaler(x.reshape(-1)) 
        
    #     d = self.output_size


    #     P = torch.zeros(list(x.shape) + [d]).to(x.device)


    #     denominator = 1 / torch.pow(self.n, 2 * torch.arange(0, d // 2).float().to(x.device) / d).unsqueeze(0)
    #     x = x.unsqueeze(1)

    #     P[:, 0::2] = torch.sin(x * denominator)
    #     P[:, 1::2] = torch.cos(x * denominator)

    #     return P

class DICE(NumeralEncoder):
    """
    This class is the DICE encoder.
    """
    def __init__(self, output_size, train_values, all_values, scaler="log"):
        super(DICE, self).__init__(output_size,train_values, all_values)

        self.M = torch.randn(self.output_size, self.output_size)
        self.Q, self.R = torch.linalg.qr(self.M)

        if torch.cuda.is_available():
            self.Q = self.Q.cuda()
            self.R = self.R.cuda()

        self.scaler_name  = scaler

        if scaler == "log":
            self.scaler = log_scaling_function
        elif scaler == "quantile":
            self.scaler = QuantileScaling(self.train_values).quantile_scaling_function

       
        
        train_value_list = sorted(self.train_values)
        
        tensor_train_values = self.scaler(torch.tensor(train_value_list))

        

        self.a = tensor_train_values[0]
        self.b = tensor_train_values[-1]


        if torch.cuda.is_available() :
            self.dice_embedding = torch.nn.Embedding(len(self.all_values), output_size).cuda()
        else:
            self.dice_embedding = torch.nn.Embedding(len(self.all_values), output_size)
        self.dice_embedding.weight.requires_grad = False


        all_values_tensor =torch.tensor(self.all_values)
        if torch.cuda.is_available() :
            all_values_tensor = all_values_tensor.cuda()

        
        x = self.scaler(all_values_tensor)

        theta = (x -self.a) / (self.b -self.a)  * np.pi

        
        # self.dice_embedding.weight = self.encode(all_values_tensor)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)


        d = torch.arange(0, self.output_size -1 ).float().to(x.device)

        self.dice_embedding.weight[:, :-1] = torch.pow(sin_theta.unsqueeze(1), d.unsqueeze(0)) * cos_theta.unsqueeze(1) 
        self.dice_embedding.weight[:, -1] = torch.pow(sin_theta, self.output_size)  


        self.dice_embedding.weight = nn.Parameter(torch.mm(self.dice_embedding.weight, self.Q))
        self.dice_embedding.weight.requires_grad = False


    def forward(self, x):
        """
        This function is the forward pass of the positional encoder.
        
        x: the input array of numbers to be encoded
        """
        if isinstance(x, list):
            x_ids = self.values2ids(x)
        else:
            x_ids =  self.values2ids([x])

        if torch.cuda.is_available() :
            x_ids = torch.tensor(x_ids).cuda()
        else:
            x_ids = torch.tensor(x_ids)
        
        return self.dice_embedding(x_ids)




    # def forward(self, x):
    #     """
    #     This function is the forward pass of the DICE encoder.
        
    #     x: the input array of numbers to be encoded
    #     """


    #     x = x.reshape(-1)
    #     seq_len = x.shape[0]
        

    #     # print("x: " + str(x))
    #     # if len(x) > 100 and self.scaler_name == "quantile":
    #     #     plt.hist(x.numpy(), bins=20)
    #     #     plt.show()

    #     # Here the random is drawn from a uniform distribution from -pi to pi, not from 0 to pi.
    #     # random_as_x = np.random.uniform(self.a - self.b, self.b, size=seq_len)
    #     # random_as_x = torch.DoubleTensor(seq_len).uniform_(self.a - self.b, self.b).to(x.device)

        
    #     x = x.double()
    #     # print("x: ", x.type())

    #     # x[x > self.b] = random_as_x[x > self.b]
    #     # x[x < self.a] = random_as_x[x < self.a]

    #     x = self.scaler(x)

    #     theta = (x -self.a) / (self.b -self.a)  * np.pi

    #     sin_theta = torch.sin(theta)
    #     cos_theta = torch.cos(theta)


    #     d = torch.arange(0, self.output_size -1 ).float().to(x.device)

    #     P = torch.zeros((seq_len, self.output_size)).to(x.device)


    #     P[:, :-1] = torch.pow(sin_theta.unsqueeze(1), d.unsqueeze(0)) * cos_theta.unsqueeze(1) 
    #     P[:, -1] = torch.pow(sin_theta, self.output_size)  

    #     projected_P = torch.mm(P, self.Q)

    #     # print("projected_P: " , P)
       
    #     return projected_P

class GMM_Prototype(NumeralEncoder):
    """
    This class is the GMM encoder.
    """
    def __init__(self, output_size, train_values, all_values, num_prototypes=20):
        super(GMM_Prototype, self).__init__(output_size, train_values, all_values)


        # train_value_list = []
        # for u in self.train_graph.nodes():
        #     if isinstance(u, float):
        #         train_value_list.append(u)

        train_value_list = self.train_values
        
        train_value_list = np.array(sorted(train_value_list))
        train_value_list = log_scaling_function_np(train_value_list)

        train_value_list = train_value_list.reshape(-1, 1)

        # print(len(train_value_list))

        print("Fitting GMM...")
        self.gmm = GaussianMixture(max_iter=10000,n_components=num_prototypes, init_params="random").fit(train_value_list)
        print("Fitting GMM... Done")


        self.weights = torch.from_numpy(self.gmm.weights_).float().view(-1).unsqueeze(0)
        self.means = torch.from_numpy(self.gmm.means_).float().view(-1).unsqueeze(0)
        self.covariances = torch.from_numpy(self.gmm.covariances_).float().view(-1).unsqueeze(0)

        self.prototype_embeddings = nn.Linear(num_prototypes, self.output_size, bias=False)
        

        # print(self.weights.shape)
        # print(self.means.shape)
        # print(self.covariances.shape)



    def forward(self, x):

        """
        This function is the forward pass of the GMM encoder.
        
        x: the input array of numbers to be encoded

        """
        x = x.reshape(-1)
        x = log_scaling_function(x)
        x = x.view(-1, 1)

        # [len(x), num_prototypes]
        similarities = self.weights.to(x.device) * torch.exp( - (x - self.means.to(x.device)) ** 2 / self.covariances.to(x.device)) 

        similarities = similarities / torch.sum(similarities, dim=1).unsqueeze(1)
    
        # [len(x), output_size]]  
        averaged_embeddings = self.prototype_embeddings(similarities)
        return averaged_embeddings

        

class DigitRNN(NumeralEncoder):
    def __init__(self, output_size, train_values, all_values):
        
        super(DigitRNN, self).__init__(output_size, train_values, all_values)

        embedding_size = output_size
        num_entities = 20

        self.unified_embeddings = nn.Embedding(num_entities, embedding_size)

        embedding_weights = self.unified_embeddings.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities, bias=False)
        self.decoder.weight = embedding_weights

        self.loss_fnc = CrossEntropyLoss()

        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size // 2, num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    
        self.string_dict = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            ".": 10,
            "-": 11,
            "[PAD]": 12
        }
        self.max_length = 32
    
    def forward(self, x):
        x = x.reshape(-1)

        batched_ids = []
        for num in x:
            num_string = str(num.item())
            
            try:
                num_ids = [self.string_dict[char] for char in num_string]
            except KeyError:
                print(num_string)
            while len(num_ids) < self.max_length:
                num_ids.append(self.string_dict["[PAD]"])

            num_ids = num_ids[:self.max_length]
            batched_ids.append(num_ids)

        batched_ids = torch.LongTensor(batched_ids)

        embeddings = self.unified_embeddings(batched_ids)

        # [batch_size, length, embedding_size] -> [batch_size, hidden_size]
        lstm_output, _ = self.LSTM(embeddings)
        lstm_output = lstm_output[:, 0, :]

        # print(lstm_output.shape)

        return lstm_output



def main():
    data_dir = "DB15K"
    experiment_number = "e34"

    print("Load Train Graph " + data_dir)
    train_path = "../preprocessing/" + data_dir +  "_train_with_units.pkl"
    train_graph = nx.read_gpickle(train_path)

    print("Load Test Graph " + data_dir)
    test_path = "../preprocessing/" + data_dir + "_test_with_units.pkl"
    test_graph = nx.read_gpickle(test_path)


    all_values = []
    for u in test_graph.nodes():
        if isinstance(u, tuple):
            all_values.append(u[0])

    train_values = []
    for u in train_graph.nodes():
        if isinstance(u, tuple):
            train_values.append(u[0])


    

    positional_encoder_log = PositionalEncoder(output_size=128, train_values=train_values, all_values=all_values, n=10000)
    dice_encoder_log = DICE(output_size=128, train_values=train_values, all_values=all_values)

    # positional_encoder_quantile = PositionalEncoder(output_size=128, train_values=train_values, all_values=all_values, n=10000, scaler="quantile")
    # dice_encoder_quantile = DICE(output_size=128, train_values=train_values, all_values=all_values, scaler="quantile")

    # lstm_encoder = DigitRNN(output_size=128, train_values=train_values, all_values=all_values)
    # gmm_encoder = GMM_Prototype(output_size=128, train_values=train_values, all_values=all_values)


    # test_sequences = torch.tensor([0.1, 1, 100, 1000, 10000, 100000, 1000000])

    test_sequences = all_values[:6]

    number_encodings = positional_encoder_log(test_sequences)
    print("number_encodings log: " + str(number_encodings.shape))
    number_encodings = positional_encoder_log.get_embeddings()
    print("number_encodings log: " + str(number_encodings.shape))

    dice_encodings = dice_encoder_log(test_sequences)
    print("dice_encodings log: " + str(dice_encodings.shape))
    dice_encodings = dice_encoder_log.get_embeddings()
    print("dice_encodings log: " + str(dice_encodings.shape))

    # number_encodings = positional_encoder_quantile(test_sequences)
    # print("number_encodings quantile: " + str(number_encodings.shape))
    # number_encodings = positional_encoder_quantile.get_embeddings()
    # print("number_encodings quantile: " + str(number_encodings.shape))

    # dice_encodings = dice_encoder_quantile(test_sequences)
    # print("dice_encodings quantile: " + str(dice_encodings.shape))
    # dice_encodings = dice_encoder_quantile.get_embeddings()
    # print("dice_encodings quantile: " + str(dice_encodings.shape))


    # lstm_encodings = lstm_encoder(test_sequences)
    # print("lstm_encodings: " + str(lstm_encodings.shape))
    # lstm_encodings = lstm_encoder.get_embeddings()
    # print("lstm_encodings: " + str(lstm_encodings.shape))


    # gmm_encodings = gmm_encoder(test_sequences)
    # print("gmm_encodings: " + str(gmm_encodings.shape))
    # gmm_encodings = gmm_encoder.get_embeddings()
    # print("gmm_encodings: " + str(gmm_encodings.shape))

if __name__ == "__main__":
    main()
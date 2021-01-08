import torch
# import config
# import utils
from src.GraphGAN import utils
from src.GraphGAN import config
from torch.nn.parameter import Parameter

# torch.set_num_threads(12)

class Discriminator():
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create tensor of shape = (self.node_emd_init.shape)
        # self.embedding_matrix = Parameter(torch.FloatTensor(self.node_emd_init.shape))
        # Assign the value to the created tensor of shape above
        self.embedding_matrix = torch.tensor(self.node_emd_init, device=self.device, requires_grad=True)

        # Create a bias tensor of shape(self.n_node,1)
        # self.bias_vector = Parameter(torch.FloatTensor(self.n_node))
        self.bias_vector = torch.zeros(self.n_node, device=self.device) 

        self.score = None
        
        # self.reward = None

        self.d_loss = None
        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.bias = None
        self.score_f = None
        self.reward = None

        self.optimizer = torch.optim.Adam([self.embedding_matrix, self.bias_vector], lr=config.lr_gen)

    def train(self, node_id, node_neighbor_id, label):
        self.node_embedding = self.embedding_matrix[node_id]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        self.bias = self.bias_vector[node_neighbor_id]
        self.bias[torch.isnan(self.bias)] = 0
        # self.score = torch.sum(torch.matmul(self.node_embedding, torch.transpose(self.node_neighbor_embedding, 0, 1))) + self.bias
        self.score = torch.sum(self.node_embedding * self.node_neighbor_embedding, 1) + self.bias
        self.score = torch.sigmoid(self.score).type(torch.FloatTensor)
        self.score = torch.cuda.FloatTensor(self.score.cuda())
        self.optimizer.zero_grad()
        loss = torch.nn.BCELoss()
        l1_loss = torch.sum(loss(self.score, torch.cuda.FloatTensor(label)))
        l2 = config.lambda_dis * (0.5 * torch.sum(self.node_neighbor_embedding ** 2) + 0.5 * torch.sum(self.node_embedding ** 2) + 0.5 * torch.sum(self.bias ** 2))
        self.d_loss = l1_loss + l2
        self.d_loss.backward()
        self.optimizer.step()

    def forward(self, node_id, node_neighbor_id):
        """ calculate reward, by using the weights that were updated after each training epoch of Discriminator"""
        self.node_embedding = self.embedding_matrix[node_id]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        self.bias = self.bias_vector[node_neighbor_id]
        self.bias[torch.isnan(self.bias)] = 0
        # self.score_f = torch.sum(torch.matmul(self.node_embedding, torch.transpose(self.node_neighbor_embedding, 0, 1))) + self.bias
        self.score_f = torch.sum(self.node_embedding * self.node_neighbor_embedding, 1) + self.bias
        self.score_f = torch.clamp(self.score_f, -10, 10)
        self.reward = torch.log(1 + torch.exp(self.score_f))
        return self.reward


# if __name__ == '__main__':
#     n_node, _ = utils.read_edges(config.train_filename, config.test_filename)
#
#     node_emd_init = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
#                                           n_node=n_node,
#                                           n_embed=config.n_emb)
#     discriminator = Discriminator(n_node, node_emd_init)
#     print("Done!!!!")

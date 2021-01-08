import torch
# import config
from src.GraphGAN import config
from src.GraphGAN import utils
from torch.nn.parameter import Parameter

class Generator():
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # shape of embedding_matrix = (n_node, n_emb) = (n_node, 50)
        # Create tensor of shape = (self.node_emd_init.shape)
        # self.embedding_matrix = Parameter(torch.FloatTensor(self.node_emd_init.shape))
        # Assign the value to the created tensor of shape above
        self.embedding_matrix = torch.tensor(self.node_emd_init,device=self.device, requires_grad=True)

        # Create a bias tensor of shape(self.n_node,1)
        # self.bias_vector = Parameter(torch.FloatTensor(self.n_node))
        self.bias_vector = torch.zeros(self.n_node, device=self.device)

        # self.all_score = None
        self.all_score = torch.matmul(self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)) + \
                         self.bias_vector # (n_node, n_node) = (5252, 5242)
        self.reward = None

        self.g_loss = None

        self.optimizer = torch.optim.Adam([self.embedding_matrix, self.bias_vector], lr=config.lr_gen)

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.bias = None
        self.score = None
        self.prob = None

        #  input data

    def train(self, node_id, node_neighbor_id, reward):

        self.reward = reward

        # look up the corresponding embedding vector for the node_id in the embedding matrix
        self.node_embedding = self.embedding_matrix[node_id]

        # look up the corresponding embedding vector for the node_neighbor_id in the embedding matrix
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]

        self.bias = self.bias_vector[node_neighbor_id]
        self.bias[torch.isnan(self.bias)] = 0

        self.score = torch.sum(self.node_embedding * self.node_neighbor_embedding, 1) + self.bias
        self.prob = torch.clamp(torch.sigmoid(self.score), 1e-5, 1)
        optimizer.zero_grad()
        l1_loss = -torch.mean(torch.log(self.prob) * self.reward)
        l2 = config.lambda_gen * (0.5 * torch.sum(
            self.node_neighbor_embedding ** 2) + 0.5 * torch.sum(
                    self.node_embedding ** 2))
        self.g_loss = l1_loss + l2
        
        self.g_loss.backward(retain_graph=True)
        self.optimizer.zero_grad()
        self.optimizer.step()

        return self.prob


# if __name__ == '__main__':
#     n_node, _ = utils.read_edges(config.train_filename, config.test_filename)
#
#     node_emd_init = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
#                                           n_node=n_node,
#                                           n_embed=config.n_emb)
#     generator = Generator(n_node, node_emd_init)
#
#     print('...:', generator.embedding_matrix)
#     print('?: ', generator.embedding_matrix.shape)
#     print("::", generator.embedding_matrix - torch.tensor(node_emd_init))
#
#     print("Done!!!!")

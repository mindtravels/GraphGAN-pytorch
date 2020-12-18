import torch
import config
import utils
from torch.nn.parameter import Parameter


class Discriminator():
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        # Create tensor of shape = (self.node_emd_init.shape)
        # self.embedding_matrix = Parameter(torch.FloatTensor(self.node_emd_init.shape))
        # Assign the value to the created tensor of shape above
        self.embedding_matrix = torch.tensor(self.node_emd_init, requires_grad=True)

        # Create a bias tensor of shape(self.n_node,1)
        # self.bias_vector = Parameter(torch.FloatTensor(self.n_node))
        self.bias_vector = torch.zeros(self.n_node)

        self.score = None
        # self.reward = None

        self.d_loss = None

    def train(self, node_id, node_neighbor_id, label):
        node_embedding = self.embedding_matrix[node_id]
        node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        bias = self.bias_vector[node_neighbor_id]
        bias[torch.isnan(bias)] = 0
        self.score = torch.sum(torch.matmul(node_embedding, torch.transpose(node_neighbor_embedding, 0, 1))) + bias
        self.score = torch.sigmoid(self.score)
        loss = torch.nn.BCELoss()
        l1_loss = torch.sum(loss(self.score, torch.FloatTensor(label)))
        l2 = config.lambda_dis * (0.5 * torch.sum(node_neighbor_embedding ** 2) + 0.5 * torch.sum(node_embedding ** 2) + 0.5 * torch.sum(bias ** 2))
        self.d_loss = l1_loss + l2
        # self.d_loss = torch.sum(
        #     loss((self.score, 2), label)) + config.lambda_dis * (
        #     0.5 * torch.sum(node_neighbor_embedding ** 2) + 0.5 *
        #     torch.sum(node_embedding ** 2) + 0.5 *
        #     torch.sum(bias ** 2))
        optimizer = torch.optim.Adam([self.embedding_matrix, self.bias_vector], lr=config.lr_gen)
        self.d_loss.backward()
        optimizer.step()
        # self.score = torch.clamp(self.score, 1e-5, 1)
        # self.reward = torch.log(1 + torch.exp(self.score))

    def forward(self, node_id, node_neighbor_id):
        """ calculate reward, by using the weights that were updated after each training epoch of Discriminator"""
        node_embedding = self.embedding_matrix[node_id]
        node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        bias = self.bias_vector[node_neighbor_id]
        bias[torch.isnan(bias)] = 0
        score = torch.sum(torch.matmul(node_embedding, torch.transpose(node_neighbor_embedding, 0, 1))) + bias
        score = torch.clamp(score, 1e-5, 1)
        reward = torch.log(1 + torch.exp(score))
        return reward


# if __name__ == '__main__':
#     n_node, _ = utils.read_edges(config.train_filename, config.test_filename)
#
#     node_emd_init = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
#                                           n_node=n_node,
#                                           n_embed=config.n_emb)
#     discriminator = Discriminator(n_node, node_emd_init)
#     print("Done!!!!")

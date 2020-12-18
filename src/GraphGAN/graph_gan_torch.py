import torch
import pickle
import numpy as np
import os
import collections
import multiprocessing
import config
import utils
import gen_torch
import dis_torch
from evaluation import link_prediction as lp
import tqdm

class GraphGan(object):
    def __init__(self):
        print("reading graphs...")
        self.n_node, self.graph = utils.read_edges(config.train_filename, config.test_filename)
        self.root_nodes = [i for i in range(self.n_node)]

        print("reading initial embeddings...")

        self.node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.trees = None
        if os.path.isfile(config.cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(config.cache_filename, 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(config.cache_filename, 'wb')
            if config.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()
        #
        # self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        # self.saver = tf.train.Saver()
        #
        # self.config = tf.ConfigProto()
        # self.config.gpu_options.allow_growth = True
        # self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess = tf.Session(config=self.config)
        # self.sess.run(self.init_op)
        # print(",,,,,,: ", self.trees)

    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            # note that nodes is an uniquely ordered set
            # tree = {0: {0 : [nb_1, nb_2, ..., nb_k],  nb_1: [0, ...]}, 1 : {1: [nb_1,...], nb_1 : [..]},...}
            trees[root] = {}
            trees[root][root] = [root]
            print('test...', trees[root][root])
            used_nodes = set()
            # queue has the form as following queue([root] for root in tqdm.tqdm(nodes)
            # with each node, we construct the tree rooted at that node, denoted as queue(['root'])
            queue = collections.deque([root]) # deque([0]) -> deque([0,1])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    # sub_node is not ordered
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_discriminator(self):
        """initializing the discriminator"""
        self.discriminator = dis_torch.Discriminator(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def build_generator(self):
        """initializing the generator"""
        self.generator = gen_torch.Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def train(self):
        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)

            # # save the model
            # if epoch > 0 and epoch % config.save_steps == 0:
            #     self.saver.save(self.sess, config.model_log + "model.checkpoint")

            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for d_epoch in range(config.n_epochs_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % config.dis_interval == 0:
                    center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                # print('center_nodes: ', len(center_nodes))
                # print('neighbor_nodes: ', len(neighbor_nodes))
                # print('labels', len(labels))
                # print("Pass prepare d")
                # training
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, config.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + config.batch_size_dis
                    # self.sess.run(self.discriminator.d_updates,
                    #               feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
                    #                          self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                    #                          self.discriminator.label: np.array(labels[start:end])})

                    self.discriminator.train(center_nodes[start:end], neighbor_nodes[start:end], labels[start:end])
                # print("pass d")

            # G-steps
            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(config.n_epochs_gen):
                if g_epoch % config.gen_interval == 0:
                    node_1, node_2, reward = self.prepare_data_for_g()
                print("Pass prepare g")

                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, config.batch_size_gen))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + config.batch_size_gen
                    # self.sess.run(self.generator.g_updates,
                    #               feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                    #                          self.generator.node_neighbor_id: np.array(node_2[start:end]),
                    #                          self.generator.reward: np.array(reward[start:end])})
                    self.generator.train(node_1[start:end], node_2[start:end], reward[start:end])
                print("pass g")

            # self.write_embeddings_to_file()
            # self.evaluation(self)
        print("training completes")

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""

        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                # self.graph[i] = [neighbors of i]
                pos = self.graph[i]
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                # print("neg is: ", neg)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        # print("cen: ", center_nodes)
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""

        paths = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
                # for each root, we generate 20 samples, each sample is equal to one path from root to that sample
        # So, we will get maximum (num_root x 20) paths
        # path is a list with length = (N x num_sample), with num_sample = 20
        # paths =[[path_root1_to_sample1],[path_root1_to_sample2],....,[path_root1_to_sample20],
        #         [path_root2_to_sample1],[path_root2_to_sample2],....,[path_root2_to sample20]
        #         .
        #         .
        #         [path_rootN_to_sample1],[path_rootN_to_sample2],....,[path_rootN_to_sample20]]
        # get_node_pairs_from_path

        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        # node_pairs = [[node pairs for path_root1_to_sample1],[node pairs for path_root1_to_sample2],....,[node pairs for path_root1_to_sample20],
        #               [node_pairs for path_root2_to_sample1],[node pairs for path_root2_to_sample2],....,[node pairs for path_root2_to sample20],
        #                .
        #                .
        #               [node pairs for path_rootN_to_sample1],[node pairs for path_rootN_to_sample2],....,[node pairs for path_rootN_to_sample20]]

        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        # reward = self.sess.run(self.discriminator.reward,
        #                        feed_dict={self.discriminator.node_id: np.array(node_1),
        #                                   self.discriminator.node_neighbor_id: np.array(node_2)})
        reward = self.discriminator.forward(node_1, node_2)
        return node_1, node_2, reward

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree

        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        # all_score = self.sess.run(self.generator.all_score)
        # all_score is a matrix with shape [n_node, n_node]
        all_score = self.generator.all_score
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                # print("////", tree[current_node])
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)

                # we retrieve embeddings corresponding to current node's neighbors
                # the multiply of g_v with shape (n_node, 1) and g_vi with shape(n_node, 1) has the shape [n_node, n_node]
                # to calculate the multiply of g_v and g_vi: we calculate the multiply of embedding_matrix with shape(n_node, 50) and its transpose
                # then saved the result in self.score with shape (n_node, n_node) in dis_torch.py
                # then we retrieve an embedding that equal to the multiply of g_v(current node) to one of its neighbor g_vi. To do that,
                # we do as following: all_score[current_id][neighbor_id]
                # due to for each current_node, we have a list of its neighbors, saved in [node_neighbor]
                # so that, we can retrieve a list of multiply of g_v and its neighbors, as the following:
                relevance_probability = all_score[current_node][node_neighbor]

                # convert tensor to numpy array
                relevance_probability = relevance_probability.detach().numpy()

                # finally, applying softmax function, we get the relevance probability of current_node and its neighbors, as formed in the paper
                relevance_probability = utils.softmax(relevance_probability)
                # print("...", node_neighbor)
                # pick a random node from its neighbors based on relevance_probability
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                # print("???", next_node)
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    # print("found!")
                    # print(previous_node, next_node)
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1 # n equal to sample_num
        return samples, paths  # for each sample, we get one path from root to that sample

    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs


if __name__ == "__main__":
    # print(config.train_filename)
    graph_gan = GraphGan()
    graph_gan.train()

















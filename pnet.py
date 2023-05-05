import torch
import torch.nn as nn

import torch.nn.functional as F
from networks import Conv4
import numpy as np

class PrototypicalNetwork(nn.Module):

    def __init__(self, num_ways, input_size, similarity="euclidean", **kwargs):
        # euclidean means that we use the negative squared Euclidean distance as kernel
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.criterion = nn.CrossEntropyLoss()

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 
        self.similarity = similarity


    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Function that applies the Prototypical network to a given task and returns the predictions 
        on the query set as well as the loss on the query set

        :param x_supp (torch.Tensor): support set images
        :param y_supp (torch.Tensor): support set labels
        :param x_query (torch.Tensor): query set images
        :param y_query (torch.Tensor): query set labels
        :param training (bool): whether we are in training mode

        :return:
          - query_preds (torch.Tensor): our predictions for all query inputs of shape [#query inputs, #classes]
          - query_loss (torch.Tensor): the cross-entropy loss between the query predictions and y_query
        """
        
        # these are the basic steps:
        # 1. compute the class centroids for the support set: m_n = 1/Xn * sum(all embeddings in the support set)
        # 2. compute pairwise similarities between query image and all class centroids
        # 3. predict class of most similar centroid 
        
        # get the embedding (g_theta) for every support image
        out_supp = self.network.forward(x_supp)


        # 1. compute the class centroids for the support set
        
        #   For every class, do the following:
        #   -retrieve every embedding in the support set for this class (class members)
        #   -calculate the class centroid: m_n = 1 / |X_n| * sum of all embeddings (slides)
        # by iterating over the class it ensures that the centroid embeddings are ordered by class 
        temp = y_supp.cpu().numpy()
        
        centroids = torch.tensor([]).to("cuda:0")
        for i in range(self.num_ways):
            # retrieve the members of the current class
            indices = np.where(temp == i)
            class_members = out_supp[indices]

            # compute the class centroid based on all the items in the support set
            class_centroid = class_members.mean(axis=0)

            #print(f"3: {centroids.get_device()}")

            # add the class centroid to the tensor with all the centroids
            centroids = torch.cat([centroids, class_centroid]) 

        #reshape the tensor in order to get the right format
        # the final shape of centroids will be: [n_ways, output.shape[1]] = [#classes, #features for embedding]
        centroids = torch.reshape(centroids, (self.num_ways, out_supp.shape[1]))
            
        # 2. compute the pairwise similarities between the query images and the class centroids 
        #retrieve the embedding of the query set
        out_query = self.network.forward(x_query)

        # retrieve euclidean distance to every centroid
        # this is actually our prediction format already
        query_preds = torch.cdist(out_query, centroids, p=2.0, compute_mode='use_mm_for_euclid_dist')

        # transform to retrieve the negative squared euclidean distance (this is the actual similarity measure)
        query_preds = -1 * (query_preds ** 2)

        # calculate the crossentropy loss
        query_loss = self.criterion(query_preds, y_query)

        if training:
            query_loss.backward() # do not remove this if statement, otherwise it won't train

        #raise NotImplementedError()
        return query_preds, query_loss

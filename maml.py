import torch
import torch.nn as nn
import higher

from networks import Conv4

class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 


    # controller input = image + label_previous
    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Performs the inner-level learning procedure of MAML: adapt to the given task 
        using the support set. It returns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters 

        :param x_supp (torch.Tensor): the support input iamges of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """

        # number of shots per class in the query set
        num_shots_query = int(len(y_query) / self.num_ways)

        # set fast_weights to the original model weights at the beginning of each task using param.clone()
        fast_weights = []
        for param in list(self.network.parameters()):
            fast_weights.append(param.clone())

        # begin iteration for the number of updates
        for t in range(self.num_updates):
            # forward the model to get the output using fast_weights
            output_supp = self.network(x_supp, weights= fast_weights)

            # get inner loss (cross-entropy loss)
            train_loss = self.inner_loss(output_supp, y_supp)

            # get the gradients w.r.t the train loss
            if self.second_order == True:
                gradients = torch.autograd.grad(train_loss, fast_weights, create_graph=True)
            else:
                gradients = torch.autograd.grad(train_loss, fast_weights)

            # compute updated weights
            for i, theta in enumerate(fast_weights):
                theta_updated = theta - self.inner_lr * gradients[i]
                fast_weights[i] = theta_updated

        # get the predictions on the query set using the updated weights
        output_query = self.network(x_query, weights= fast_weights)

        # compute the outer loss (i.e. compute L(f(theta_prime))) on the query set
        query_loss = self.inner_loss(output_query, y_query)
        query_preds = output_query.view(self.num_ways * num_shots_query, self.num_ways)

        if training:
            query_loss.backward() 

        return query_preds, query_loss


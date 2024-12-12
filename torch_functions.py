import numpy as np
import torch
import torchvision



def pairwise_distances(x, y=None):
    """
    Compute pairwise distances between points.
    Input: 
        x: N x d tensor
        y: Optional M x d tensor
    Output: 
        NxM matrix where dist[i, j] is the squared distance between x[i, :] and y[j, :]
        If y is not provided, compute distances within x.
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class MyTripletLossFunc(torch.autograd.Function):
    """
    Custom autograd Function for triplet loss.
    """

    @staticmethod
    def forward(ctx, features, triplets):
        """
        Forward pass for the triplet loss.
        """
        ctx.save_for_backward(features)
        ctx.triplets = triplets
        ctx.triplet_count = len(triplets)

        distances = pairwise_distances(features).cpu().numpy()

        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in triplets:
            w = 1.0
            triplet_count += w
            loss += w * np.log(1 + np.exp(distances[i, j] - distances[i, k]))
            if distances[i, j] < distances[i, k]:
                correct_count += 1

        ctx.distances = distances
        loss /= triplet_count
        return torch.FloatTensor([loss])

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the triplet loss.
        """
        features, = ctx.saved_tensors
        features_np = features.cpu().numpy()
        grad_features = features.clone().zero_()
        grad_features_np = grad_features.cpu().numpy()

        for i, j, k in ctx.triplets:
            w = 1.0
            f = 1.0 - 1.0 / (
                1.0 + np.exp(ctx.distances[i, j] - ctx.distances[i, k]))
            grad_features_np[i, :] += w * f * (features_np[i, :] - features_np[j, :]) / ctx.triplet_count
            grad_features_np[j, :] += w * f * (features_np[j, :] - features_np[i, :]) / ctx.triplet_count
            grad_features_np[i, :] += -w * f * (features_np[i, :] - features_np[k, :]) / ctx.triplet_count
            grad_features_np[k, :] += -w * f * (features_np[k, :] - features_np[i, :]) / ctx.triplet_count

        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= float(grad_output[0])
        return grad_features, None


class TripletLoss(torch.nn.Module):
    """
    Wrapper class for the triplet loss.
    """

    def __init__(self, pre_layer=None):
        super(TripletLoss, self).__init__()
        self.pre_layer = pre_layer

    def forward(self, x, triplets):
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        loss = MyTripletLossFunc.apply(x, triplets)
        return loss


class NormalizationLayer(torch.nn.Module):
    """
    Normalization layer for scaling features.
    """

    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor([self.norm_s]))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features

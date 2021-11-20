import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num*2, 0)
        center = []
        for i in range(label_num*2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



            
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

# 两模态异中心加权正则化三元组损失
class TripletLoss_CWRT(nn.Module):
    """Weighted Regularized Triplet'."""
    """ Hetero-center-Weighted Regularized Triplet-loss-for-VT-Re-ID
       "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
       [(arxiv)](https://arxiv.org/abs/2008.06223).

        Args:
        - margin (float): margin for triplet.
        """

    def __init__(self):
        super(TripletLoss_CWRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels, normalize_feature=False):    # torch.Size([128, 2048])  128
        if normalize_feature:
            feats = normalize(feats, axis=-1)

        label_uni = labels.unique()                   # 8 unique函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
        targets = torch.cat([label_uni, label_uni])   # 16
        label_num = len(label_uni)                    # 8
        feat = feats.chunk(label_num * 2, 0)          # 16  torch.Size([8, 2048])   将数组分组
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))    # 求取每张图片的均值 16个均值
        inputs = torch.cat(center)


        dist_mat = pdist_torch(inputs, inputs)
        N = dist_mat.size(0)                                           # 16  # torch.Size([16, 16]) up
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()   # torch.Size([16, 16])
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()   # torch.Size([16, 16])

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos                                          # torch.Size([16, 16])
        dist_an = dist_mat * is_neg                                          # torch.Size([16, 16])

        weights_ap = softmax_weights(dist_ap, is_pos)         # torch.Size([16, 16])
        weights_an = softmax_weights(-dist_an, is_neg)        # torch.Size([16, 16])
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)       # 16
        closest_negative = torch.sum(dist_an * weights_an, dim=1)        # 16

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)   # 16
        loss = self.ranking_loss(closest_negative - furthest_positive, y)    # tensor(6.6254, device='cuda:0', grad_fn=<SoftMarginLossBackward>)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()  # 0
        return loss, correct

# 三模态异中心加权正则化三元组损失
class TripletLoss_TCWRT(nn.Module):
    """Weighted Regularized Triplet'."""
    """ Hetero-center-Weighted Regularized Triplet-loss-for-VT-Re-ID
       "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
       [(arxiv)](https://arxiv.org/abs/2008.06223).

        Args:
        - margin (float): margin for triplet.
        """

    def __init__(self):
        super(TripletLoss_TCWRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels, normalize_feature=False):  # torch.Size([12, 10])  12 tensor([ 52,  52, 182, 182,  52,  52, 182, 182,  52,  52, 182, 182], dtype=torch.int32)
        if normalize_feature:
            feats = normalize(feats, axis=-1)

        label_uni = labels.unique()                             # 2 unique函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
        targets = torch.cat([label_uni, label_uni, label_uni])  # 6 类别数*模态数 tensor([ 52, 182,  52, 182,  52, 182], dtype=torch.int32)
        label_num = len(label_uni)                              # 2
        feat = feats.chunk(label_num * 3, 0)                    # 6  torch.Size([2, 10])   将数组分组，分6组，第一组表示可见光模态的第一类图片（前两张图片），第三组表示中间模态的第一类图片（第56张图片）
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))  # 分别求取每个模态中每种类别的中心，3个模态，2类，共求取6个中心
        inputs = torch.cat(center)                              # torch.Size([6, 10])

        dist_mat = pdist_torch(inputs, inputs)
        N = dist_mat.size(0)  # 4  torch.Size([4, 4]) up
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()  # 相同的类中心为1 torch.Size([4, 4])
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()  # 不同的类中心为1 torch.Size([4, 4])

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos  # torch.Size([16, 16])
        dist_an = dist_mat * is_neg  # torch.Size([16, 16])

        weights_ap = softmax_weights(dist_ap, is_pos)  # torch.Size([16, 16])
        weights_an = softmax_weights(-dist_an, is_neg)  # torch.Size([16, 16])
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)  # 16
        closest_negative = torch.sum(dist_an * weights_an, dim=1)  # 16

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)  # 16
        loss = self.ranking_loss(closest_negative - furthest_positive,
                                 y)  # tensor(6.6254, device='cuda:0', grad_fn=<SoftMarginLossBackward>)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()  # 0
        return loss, correct

# 加权正则化三元组损失
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

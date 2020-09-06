import torch

def cosine_distance(x1, x2=None, eps=1e-8):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    if x2 is None:
        distances = 1 - torch.mm(x1, x1.t()) / (w1 * w1.t()).clamp(min=eps)
    else:
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        distances = 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    # assert distances[distances<-eps].shape[0] == 0
    # assert distances[distances>2+eps].shape[0] == 0
    distances = distances.clamp(min = 0, max=2)
    return distances

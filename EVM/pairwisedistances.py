import torch

def cosine(x, y, unit_vectors=False):
    print(" WARNING: COSINE DISTANCE RESULTS HAVE NOT BEEN VERIFIED".center(90, '*'))
    if not unit_vectors:
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
    similarity = torch.einsum('nc,ck->nk', [x, y.T])
    distances = 1-similarity
    return distances

def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    return distances

__dict__ = {'cosine':cosine,
            'euclidean':euclidean
           }
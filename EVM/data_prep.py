import logging
import time
import math
import h5py
import torch

logger = logging.getLogger("data-prep")

def read_features(args, cls_to_process):
    try:
        h5_objs = [h5py.File(file_name, "r") for file_name in args.feature_files]
        file_layer_comb = list(zip(h5_objs, args.layer_names))
        for cls in cls_to_process:
            temp = []
            for hf, layer_name in file_layer_comb:
                temp.append(torch.squeeze(torch.tensor(hf[cls][layer_name])))
            features = torch.cat(temp)
            yield features
    finally:
        for h in h5_objs:
            h.close()

def prep_all_features(gpu, args):
    startTime = time.time()
    # total number of classes
    l = len(args.all_classes)
    # batches of negative classes
    c = list(range(0, l + l % args.world_size + 1, math.ceil(l / args.world_size)))
    cls_to_process_ = args.all_classes[c[gpu] : c[gpu + 1]]
    cls_to_process, cls_to_process_shapes = zip(*cls_to_process_)
    features_gen = read_features(args, cls_to_process)
    features = []
    start = 0
    start_indxs = []
    for feature in features_gen:
        start_indxs.append(start)
        features.append(feature)
        start += feature.shape[0]
    start_indxs.append(start)
    features = torch.cat(features).to(f"cuda:{gpu}")
    logger.info(
        f"***************** Time taken to load/prepare data *****************{time.time() - startTime}"
    )
    return features, start_indxs
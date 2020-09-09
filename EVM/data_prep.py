import time
import math
import h5py
import torch
from tqdm import tqdm
# import tables

def read_features(args, cls_to_process=None):
    try:
        # h5_objs = [tables.open_file(file_name, driver="H5FD_CORE") for file_name in args.feature_files]
        h5_objs = [h5py.File(file_name, "r") for file_name in args.feature_files]
        file_layer_comb = list(zip(h5_objs, args.layer_names))
        if cls_to_process is None:
            cls_to_process = sorted(list(h5_objs[0].keys()))
        if args.debug:
            cls_to_process = cls_to_process[:50]
        for cls in cls_to_process:
            temp = []
            for hf, layer_name in file_layer_comb:
                temp.append(torch.squeeze(torch.tensor(hf[cls][layer_name])))
            features = torch.cat(temp)
            yield cls,features
    finally:
        for h in h5_objs:
            h.close()


def add_chunk(all_classes,per_chunk_features_t,starting_indxs_dict):
    per_chunk_features_t = torch.cat(per_chunk_features_t)
    per_chunk_features_norm = per_chunk_features_t.norm(p=2, dim=1, keepdim=True)
    all_classes['features_t'].append(per_chunk_features_t.t())
    all_classes['features_t_norm'].append(per_chunk_features_norm.t())
    all_classes['start_indxs'].append(starting_indxs_dict)
    return all_classes, 0,{},[]


def prep_all_features(args):
    features_gen = read_features(args)
    all_classes={}
    all_classes['features_t']=[]
    all_classes['features_t_norm']=[]
    all_classes['start_indxs']=[]
    start = 0
    per_chunk_features_t = []
    starting_indxs_dict = {}
    class_names = []
    pbar = tqdm(total=1000)
    for cls,feature in features_gen:
        pbar.update(1)
        per_chunk_features_t.append(feature)
        starting_indxs_dict[cls] = start
        start += feature.shape[0]
        class_names.append(cls)
        if len(per_chunk_features_t)%args.cls_per_chunk == 0:
            all_classes, start, starting_indxs_dict, per_chunk_features_t = add_chunk(all_classes,
                                                                                      per_chunk_features_t,
                                                                                      starting_indxs_dict)
    # Add any additional values missed
    if len(per_chunk_features_t)>0:
        all_classes, _, _, _ = add_chunk(all_classes,
                                         per_chunk_features_t,
                                         starting_indxs_dict)
    pbar.close()
    return all_classes, class_names
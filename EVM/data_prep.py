import time
import math
import h5py
import torch
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
# import tables

def read_features(args, feature_file_names=None, cls_to_process=None):
    try:
        # h5_objs = [tables.open_file(file_name, driver="H5FD_CORE") for file_name in feature_file_names]
        h5_objs = [h5py.File(file_name, "r") for file_name in feature_file_names]
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


def add_chunk(all_classes,per_chunk_features,starting_indxs_dict):
    per_chunk_features = torch.cat(per_chunk_features)
    per_chunk_features_norm = per_chunk_features.norm(p=2, dim=1, keepdim=True)
    all_classes['features_t'].append(per_chunk_features.t())
    all_classes['features_t_norm'].append(per_chunk_features_norm.t())
    all_classes['start_indxs'].append(starting_indxs_dict)
    return all_classes, 0,{},[]


def prep_all_features(args):
    features_gen = read_features(args, feature_file_names = args.feature_files)
    all_classes={}
    all_classes['features_t']=[]
    all_classes['features_t_norm']=[]
    all_classes['start_indxs']=[]
    start = 0
    per_chunk_features = []
    starting_indxs_dict = {}
    class_names = []
    pbar = tqdm(total=1000)
    for cls,feature in features_gen:
        pbar.update(1)
        per_chunk_features.append(feature)
        starting_indxs_dict[cls] = (start, start+feature.shape[0])
        start += feature.shape[0]
        class_names.append(cls)
        if len(per_chunk_features)%args.cls_per_chunk == 0:
            all_classes, start, starting_indxs_dict, per_chunk_features_t = add_chunk(all_classes,
                                                                                      per_chunk_features,
                                                                                      starting_indxs_dict)
    # Add any additional values missed
    if len(per_chunk_features)>0:
        all_classes, _, _, _ = add_chunk(all_classes,
                                         per_chunk_features,
                                         starting_indxs_dict)
    pbar.close()
    return all_classes, class_names


def prep_single_chunk(args, cls_to_process):
    features_gen = read_features(args,
                                 feature_file_names = args.feature_files,
                                 cls_to_process = cls_to_process)
    all_classes={}
    all_classes['features_t']=[]
    all_classes['features_t_norm']=[]
    all_classes['start_indxs']=[]
    start = 0
    per_chunk_features = []
    starting_indxs_dict = {}
    class_names = []
    for cls, feature in features_gen:
        per_chunk_features.append(feature)
        starting_indxs_dict[cls] = (start, start+feature.shape[0])
        start += feature.shape[0]
        class_names.append(cls)
    all_classes, _, _, _ = add_chunk(all_classes,
                                     per_chunk_features,
                                     starting_indxs_dict)
    return all_classes, class_names


def prep_all_features_parallel(args):
    start_time = time.time()
    with h5py.File(args.feature_files[0], "r") as hf:
        all_class_names = sorted(list(hf.keys()))
    if args.debug:
        all_class_names = all_class_names[:100]
    all_class_batches = [all_class_names[i:i+args.cls_per_chunk] for i in range(0, len(all_class_names), args.cls_per_chunk)]
    p = mp.Pool(len(all_class_batches))
    all_data = p.map(partial(prep_single_chunk, args), all_class_batches)
    all_classes={}
    all_classes['features_t']=[]
    all_classes['features_t_norm']=[]
    all_classes['start_indxs']=[]
    all_class_names = []
    for classes_meta, class_names in all_data:
        all_class_names.extend(class_names)
        all_classes['features_t'].extend(classes_meta['features_t'])
        all_classes['features_t_norm'].extend(classes_meta['features_t_norm'])
        all_classes['start_indxs'].extend(classes_meta['start_indxs'])
    print(f"Finished feature reading in {time.time() - start_time} seconds")
    return all_classes, all_class_names
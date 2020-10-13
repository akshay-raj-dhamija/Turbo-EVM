import h5py
import math
import time
from tqdm import tqdm
from EVM import data_prep
import torch
import torch.multiprocessing as mp
from functools import partial

global hf
global pbar
global count

def initializer(args, combination_dict, display_progress = False):
    global hf
    global pbar
    global count
    count = 0
    pbar = None

    evm_file_name = f"EVM_model_tail_{combination_dict['tailsize']}" \
                    f"_ct_{combination_dict['cover_threshold']}" \
                    f"_dm_{combination_dict['distance_multiplier']}"
    output_file_path = f"{args.output_path}/{evm_file_name}.hdf5"
    print(f"SAVING FILE TO {output_file_path}\n")
    hf = h5py.File(output_file_path, "w")

    if display_progress:
        pbar = tqdm(total=args.total_no_of_classes)
    return

def close(display_progress=False):
    global hf
    global pbar
    hf.close()
    if display_progress:
        pbar.close()
    return

def save_cls_evs(cls_name, extreme_vectors, extreme_vectors_indexes, covered_vectors):
    global hf
    global pbar
    global count
    g = hf.create_group(cls_name)
    w = g.create_group("weibull")
    w.create_dataset('Scale', data=extreme_vectors['Scale'])
    w.create_dataset('Shape', data=extreme_vectors['Shape'])
    w.create_dataset('signTensor', data=extreme_vectors['signTensor'])
    w.create_dataset('translateAmountTensor', data=extreme_vectors['translateAmountTensor'])
    w.create_dataset('smallScoreTensor', data=extreme_vectors['smallScoreTensor'])
    g.create_dataset('extreme_vectors_indexes', data=extreme_vectors_indexes)
    c = g.create_group("covered_vectors")
    for ind, covered_vector in enumerate(covered_vectors):
        c.create_dataset(str(ind), data=covered_vector)
    if pbar is not None:
        pbar.update(1)
    count+=1

def cls_counter():
    return count

def ev_reader(args, evm_param_combination, classes_in_batch):
    evm_model_file = f"{args.output_path}/EVM_model_tail_{evm_param_combination['tailsize']}_ct_{evm_param_combination['cover_threshold']}_dm_{evm_param_combination['distance_multiplier']}.hdf5"
    with h5py.File(evm_model_file, "r") as hf:
        for cls in classes_in_batch:
            weibull={}
            for param in hf[cls]['weibull']:
                weibull[param] = hf[cls]['weibull'][param][()]
            yield hf[cls]['extreme_vectors_indexes'][()], weibull

def load_ev_feature(args, evm_param_combination, cls_to_process):
    ev_gen = ev_reader(args, evm_param_combination, cls_to_process)
    features_gen = data_prep.read_features(args,
                                           feature_file_names=args.feature_files,
                                           cls_to_process=cls_to_process)
    to_return = []
    for cls, feature in features_gen:
        ev_indx, weibull = next(ev_gen)
        to_return.append((cls, feature[ev_indx,:].tolist(), weibull))
    return to_return

def convert_to_batches(chunks, classes_per_batch):
    """
    Takes a list of lists as input where each lowest level list contains information about
    the extreme vectors in one class
    :param args:
    :param classes_in_batch:
    :return:
    """

    def initialize_per_batch_dict(current_batch=None):
        if current_batch is not None:
            current_batch['features'] = torch.cat(current_batch['features']).double()
            current_batch['weibulls']['Scale'] = torch.cat(current_batch['weibulls']['Scale'])
            current_batch['weibulls']['Shape'] = torch.cat(current_batch['weibulls']['Shape'])
            current_batch['weibulls']['smallScoreTensor'] = torch.cat(current_batch['weibulls']['smallScoreTensor'])
            current_batch['weibulls']['signTensor'] = list(set(current_batch['weibulls']['signTensor']))[0].item()
            current_batch['weibulls']['translateAmountTensor'] = \
                list(set(current_batch['weibulls']['translateAmountTensor']))[0].item()
            old_batch = current_batch
        else:
            old_batch = None
        new_batch = {}
        new_batch['features'] = []
        new_batch['start_indx'] = {}
        new_batch['weibulls'] = {}
        new_batch['weibulls']['Scale'] = []
        new_batch['weibulls']['Shape'] = []
        new_batch['weibulls']['smallScoreTensor'] = []
        new_batch['weibulls']['signTensor'] = []
        new_batch['weibulls']['translateAmountTensor'] = []
        start = 0
        return old_batch,start,new_batch

    batches = []
    _, start, current_batch = initialize_per_batch_dict()

    for single_chunk in chunks:
        for cls, feature, weibull in single_chunk:
            feature = torch.tensor(feature)
            current_batch['start_indx'][cls] = (start, start + feature.shape[0])
            current_batch['features'].append(feature)
            [current_batch['weibulls'][_].append(torch.tensor(weibull[_])) for _ in weibull]
            start += feature.shape[0]
            if len(current_batch['features'])==classes_per_batch:
                old_batch, start, current_batch = initialize_per_batch_dict(current_batch)
                batches.append(old_batch)
    if start!=0:
        old_batch, _, _ = initialize_per_batch_dict(current_batch)
        batches.append(old_batch)
    return batches

def model_loader(args, evm_param_combination):
    """
    The function would return a list of batches containing the respective features and weibulls for all classes in a batch
    :param args:
    :return:
    """
    no_of_batches = 5
    start_time = time.time()

    # Get all feature vectors for EV's loaded
    evm_model_file = f'{args.output_path}/EVM_model_tail_{evm_param_combination.tailsize}_ct_{evm_param_combination.cover_threshold}_dm_{evm_param_combination.distance_multiplier}.hdf5'
    print(f"Loading model from {evm_model_file}")
    with h5py.File(evm_model_file, "r") as hf:
        all_class_names = list(hf.keys())

    no_of_chunks = mp.cpu_count()
    classes_per_batch = math.ceil(len(all_class_names)/no_of_chunks)
    all_class_batches = [all_class_names[i:i+classes_per_batch] for i in range(0, len(all_class_names), classes_per_batch)]
    p = mp.Pool(min(no_of_chunks,len(all_class_batches)))
    chunks = p.map(partial(load_ev_feature,args, evm_param_combination._asdict()),all_class_batches)

    classes_per_batch = math.ceil(len(all_class_names)/no_of_batches)
    batches = convert_to_batches(chunks, classes_per_batch)
    print(f"EVM model loaded in {time.time() - start_time} seconds")
    return batches
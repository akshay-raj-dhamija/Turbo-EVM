import math
import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import EVM
from EVM import data_prep
from EVM import pairwisedistances
from EVM import model_saver
from DistributionModels import weibull

def fit_weibull(distances, distance_multiplier, tailsize, gpu):
    if tailsize>distances.shape[1]:
        print(f"WARNING: tailsize {tailsize} is more than the number of samples {distances.shape}, setting it to all samples {distances.shape[1]}")
        tailsize = distances.shape[1]
    mr = weibull.weibull()
    mr.FitLow(distances.double() * distance_multiplier, tailsize, isSorted=False, gpu=gpu)
    return mr

def set_cover(mr_model, positive_distances, cover_threshold):
    # compute probabilities
    probabilities = mr_model.wscore(positive_distances)

    # threshold by cover threshold
    e = torch.eye(probabilities.shape[0]).type(torch.BoolTensor)
    thresholded = probabilities >= cover_threshold
    thresholded[e] = True
    del probabilities
    # try:
    #     assert torch.all(thresholded[e])==True, "All samples are not covering themselves after cover threshold"
    # except:
        # ForkedPdb().set_trace()

    # greedily add points that cover most of the others
    covered = torch.zeros(thresholded.shape[0]).type(torch.bool)
    extreme_vectors = []
    covered_vectors = []

    while not torch.all(covered).item():
        sorted_indices = torch.topk(torch.sum(thresholded[:, ~covered], dim=1),
                                    len(extreme_vectors)+1,
                                    sorted=False,
                                    ).indices
        for indx, sortedInd in enumerate(sorted_indices.tolist()):
            if sortedInd not in extreme_vectors:
                break
        else:
            print(thresholded.device,"ENTERING INFINITE LOOP ... EXITING")
            break
        covered_by_current_ev = torch.nonzero(thresholded[sortedInd, :], as_tuple=False)
        covered[covered_by_current_ev] = True
        extreme_vectors.append(sortedInd)
        covered_vectors.append(covered_by_current_ev.to("cpu"))
    del covered
    extreme_vectors_indexes = torch.tensor(extreme_vectors)
    params = mr_model.return_all_parameters()
    scale = torch.gather(params["Scale"].to("cpu"), 0, extreme_vectors_indexes)
    shape = torch.gather(params["Shape"].to("cpu"), 0, extreme_vectors_indexes)
    smallScore = torch.gather(
        params["smallScoreTensor"][:, 0].to("cpu"), 0, extreme_vectors_indexes
    )
    extreme_vectors_models = dict(Scale=scale,
                                  Shape=shape,
                                  signTensor=params["signTensor"],
                                  translateAmountTensor=params["translateAmountTensor"],
                                  smallScoreTensor=smallScore)
    del params
    return (extreme_vectors_models, extreme_vectors_indexes, covered_vectors)


def saver_process_execution(global_rank, args, combination_name, combination_dict, display_progress = False):
    # export GLOO_SOCKET_IFNAME=WHATEVER IN IFCONFIG (like eth0 or something
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = args.dist_url
        os.environ['MASTER_PORT'] = '9451'
        rpc.init_rpc(f"{args.saver_process_mapping[combination_name]}", rank=global_rank, world_size=args.world_size+len(args.saver_process_mapping.keys()),
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=args.world_size*3,
                                                                           rpc_timeout=0)
                     )
        print(f"Started RPC for saver process ID {global_rank} for combination {combination_name}")

    model_saver.initializer(args, combination_dict, display_progress)
    print(f"Initialized Model Saver in the Saver process ID {global_rank} ... going to wait now")
    while True:
        if model_saver.cls_counter()>=args.total_no_of_classes:
            break
    print(f"Closing Model file in the saver process ID {global_rank}")
    model_saver.close(display_progress)
    print(f"Shutting down RPC for saver process ID {global_rank}")
    rpc.shutdown()
    return

def each_process_trainer(gpu, args, pos_classes_to_process, all_class_features_meta):
    global_rank = sum(args.gpus[:args.local_rank]) + gpu
    # export GLOO_SOCKET_IFNAME=WHATEVER IN IFCONFIG (like eth0 or something
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = args.dist_url
        os.environ['MASTER_PORT'] = '9451'
        rpc.init_rpc(f"{global_rank}", rank=global_rank, world_size=args.world_size+len(args.saver_process_mapping.keys()),
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=args.world_size*3,
                                                                           rpc_timeout=0)
                     )
        print(f"Started RPC for internal process ID {global_rank}")

    torch.cuda.set_device(gpu)

    max_tailsize = max(args.tailsizes)
    for pos_cls_name in pos_classes_to_process:

        # Find positive class features
        pos_batch_no = 0
        while pos_cls_name not in all_class_features_meta['start_indxs'][pos_batch_no]:
            pos_batch_no+=1
        start, end = all_class_features_meta['start_indxs'][pos_batch_no][pos_cls_name]
        positive_cls_feature = all_class_features_meta['features_t'][pos_batch_no][:,start:end]
        positive_cls_feature = positive_cls_feature.t()
        positive_cls_feature = positive_cls_feature.to(f"cuda:{gpu}")

        negative_distances=[]
        for batch_no, neg_features in enumerate(all_class_features_meta['features_t']):
            norm = all_class_features_meta['features_t_norm'][batch_no].to(f"cuda:{gpu}")
            # distances is a tensor of size no_of_positive_samples X no_of_samples_in_current_batch
            distances = pairwisedistances.cosine_distance(positive_cls_feature,
                                                          neg_features.to(f"cuda:{gpu}"),
                                                          w2_t=norm)
            del norm
            if pos_cls_name in all_class_features_meta['start_indxs'][batch_no]:
                start, end = all_class_features_meta['start_indxs'][batch_no][pos_cls_name]
                # store the positive distances and use only negatives for tail
                positive_distances = distances[:, start : start+positive_cls_feature.shape[0]]
                positive_distances = positive_distances.cpu()
                # negative distances
                distances = torch.cat([distances[:, :start],
                                       distances[:, start+positive_cls_feature.shape[0]:]], dim=1)
                # check if distances to self is zero
                e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
                assert torch.allclose(positive_distances[e].type(torch.FloatTensor), \
                                      torch.zeros(positive_distances.shape[0]))==True, \
                    "Distances of samples to themselves is not zero"
            # Store bottom k distances from each batch to the cpu
            sortedTensor = torch.topk(distances,
                                      min(max_tailsize,distances.shape[1]),
                                      dim = 1,
                                      largest = False,
                                      sorted = True).values
            del distances
            negative_distances.append(sortedTensor.cpu())

        # For all batches iterate batch by batch to find bottom k distances
        # This is done because the gpu memory may not be able to accommodate all distances at once
        # Also, if performed on cpu it may cause a bottle neck especially for machines
        # with high number of gpus and relatively low number of cpus.
        sortedTensor = negative_distances[0].to(f"cuda:{gpu}")
        distance_batch = None
        for n in negative_distances[1:]:
            distance_batch = torch.cat([sortedTensor,n.to(f"cuda:{gpu}")], dim=1)
            sortedTensor = torch.topk(distance_batch,
                                      min(max_tailsize,distance_batch.shape[1]),
                                      dim=1,
                                      largest=False,
                                      sorted=True).values
        del distance_batch
        del negative_distances
        del neg_features
        del positive_cls_feature

        # Perform actual EVM training
        for combination in args.evm_param_combinations:
            weibull_model = fit_weibull(sortedTensor,
                                        combination.distance_multiplier, combination.tailsize, gpu)
            extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                         positive_distances.to(f"cuda:{gpu}"),
                                                                                         combination.cover_threshold)
            del weibull_model

            # Send the computed information for the extreme vectors of the current class to
            # the saver process for this specific EVM parameter combination
            _ = rpc.remote(f"{args.saver_process_mapping[combination.__str__()]}",
                           model_saver.save_cls_evs,
                           timeout=0,
                           args=(pos_cls_name,
                                 extreme_vectors_models,
                                 extreme_vectors_indexes,
                                 covered_vectors))

    if args.world_size > 1:
        print(f"Shutting down RPC for internal process ID {global_rank}")
        rpc.shutdown()

def main(args):
    # Load features for all classes
    all_class_features_meta, class_names = data_prep.prep_all_features_parallel(args)

    args.evm_param_combinations, args.saver_process_mapping = EVM.get_all_evm_combinations(args)
    args.total_no_of_classes = len(class_names)
    class_names = sorted(class_names)
    no_of_pos_cls_each_gpu = math.ceil(len(class_names)/args.world_size)
    if args.debug:
        no_of_pos_cls_each_gpu = 2
        class_names = class_names[:no_of_pos_cls_each_gpu*args.world_size]

    classes_to_process_on_each_gpu = []
    for indx in range(0,len(class_names),no_of_pos_cls_each_gpu):
        classes_to_process_on_each_gpu.append(class_names[indx:indx+no_of_pos_cls_each_gpu])

    if args.world_size == 1:
        each_process_trainer(0, args, classes_to_process_on_each_gpu[0], all_class_features_meta)
    else:
        processes = []
        for rank in range(args.gpus[args.local_rank]):
            p = mp.Process(target=each_process_trainer,
                           args=(rank, args,
                                 classes_to_process_on_each_gpu[rank],
                                 all_class_features_meta))
            p.start()
            processes.append(p)
        # Start the saver process only on the last node
        if args.local_rank==len(args.gpus)-1:
            for i, evm_param_combination_tuple in enumerate(args.evm_param_combinations):
                saver_process = mp.Process(target=saver_process_execution,
                                           args=(args.saver_process_mapping[evm_param_combination_tuple.__str__()], args,
                                                 evm_param_combination_tuple.__str__(),
                                                 evm_param_combination_tuple._asdict(),
                                                 True if i==0 else False))
                saver_process.start()
                processes.append(saver_process)

    if args.world_size > 1:
        for p in processes:
            p.join()



import torch
from .utils.debugger import ForkedPdb
import torch.distributed.rpc as rpc
import time
import h5py
import math
from .data_prep import prep_all_features
from EVM import pairwisedistances
from DistributionModels import weibull

def fit_weibull(distances, distance_multiplier, tailsize, gpu):
    if tailsize>distances.shape[1]:
        print("WARNING: tailsize is more than the number of sample, setting it to all samples")
        tailsize = distances.shape[1]
    mr = weibull.weibull()
    mr.FitLow(distances.double() * distance_multiplier, tailsize, isSorted=False, gpu=gpu)
    return mr

def set_cover(mr_model, positive_distances, cover_threshold):
    # compute probabilities
    probabilities = mr_model.wscore(positive_distances)
    del positive_distances

    # threshold by cover threshold
    e = torch.eye(probabilities.shape[0]).type(torch.BoolTensor)
    thresholded = probabilities >= cover_threshold
    del probabilities
    assert torch.all(thresholded[e]) == True, "All samples are not covering themselves after cover threshold"
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



def trainer(gpu, args, to_save_queue):
    if args.world_size > 1:
        torch.distributed.init_process_group(backend="gloo",
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=sum(args.gpus[:args.local_rank])+gpu)
    torch.cuda.set_device(gpu)
    all_features, start_indxs = prep_all_features(gpu, args)
    no_of_batches = math.ceil(len(args.all_classes) / args.world_size)
    processing_order = [
                        list(range(s, len(args.all_classes), no_of_batches))
                        for s in range(no_of_batches)
                    ]
    cls_to_process = [args.all_classes[i] for p in processing_order for i in p]
    no_of_classes_processed = 0
    for batch_no, batch in enumerate(processing_order):
        distances_to_process = None
        for i, pos_cls_no in enumerate(batch):
            current_cls_being_processed, current_cls_shape = cls_to_process[no_of_classes_processed]
            no_of_classes_processed+=1
            if i == gpu:
                positive_cls_feature = all_features[start_indxs[batch_no]:start_indxs[batch_no + 1]]
            else:
                positive_cls_feature = torch.zeros(current_cls_shape).type(all_features.dtype).to(all_features.device)

            if args.world_size != 1:
                _ = torch.distributed.broadcast(positive_cls_feature, i)
                del _

            # distances is a tensor of size no_of_positive_features X no_of_features_on_gpu
            distances = pairwisedistances.cosine_distance(positive_cls_feature,
                                                                     all_features)
            del positive_cls_feature
            # sorting the distances on each gpu ... make sure the shape of sortedTensor is the same for all gpus
            assert distances.shape[1]>=args.tailsize, \
                "ERROR: The tailsize should be less/equal to the number of negative samples on each gpu, needed for gather"
            if gpu == i:
                current_cls_being_processed_by_this_gpu = current_cls_being_processed
                ### store the positive distances and use only negatives for tail
                positive_distances = distances[:, start_indxs[batch_no] : start_indxs[batch_no + 1]]
                negative_distances = torch.cat([distances[:, :start_indxs[batch_no]],
                                                distances[:, start_indxs[batch_no+1]:]], dim=1)
                del distances
                distances = negative_distances
                # check if distances to self is zero
                e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
                assert torch.allclose(positive_distances[e].type(torch.FloatTensor), \
                                       torch.zeros(positive_distances.shape[0]))==True, \
                        "Distances of samples to themselves is not zero"

            sortedTensor = torch.topk(distances,
                                      args.tailsize,
                                      dim=1,
                                      largest=True,
                                      sorted=True).values

            if gpu == i:
                ### Gather all the distances from all gpus
                if args.world_size == 1:
                    distances_to_process = sortedTensor
                else:
                    distances_to_process = [torch.zeros_like(sortedTensor) for _ in range(args.world_size)]
                    _ = torch.distributed.gather(tensor=sortedTensor, dst=gpu, gather_list=distances_to_process)
                    distances_to_process = torch.cat(distances_to_process, dim=1)
                    del _
                distances_to_process = torch.topk(distances_to_process,
                                                  min(args.tailsize, distances_to_process.shape[1]),
                                                  dim=1,
                                                  largest=True,
                                                  sorted=True).values
            else:
                pl = torch.distributed.gather(tensor=sortedTensor, dst=i)
                del pl
            # torch.distributed.barrier()

        # process the part only specific to your gpu
        if distances_to_process is not None:
            weibull_model = fit_weibull(distances_to_process, args.distance_multiplier, args.tailsize, gpu)
            extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                         positive_distances,
                                                                                         args.cover_threshold)
            to_save_queue.put((current_cls_being_processed_by_this_gpu,
                               extreme_vectors_models,
                               extreme_vectors_indexes,
                               covered_vectors))
            del weibull_model
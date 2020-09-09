import math
import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from EVM import pairwisedistances
from EVM import data_prep
from EVM import model_saver
from .utils.debugger import ForkedPdb
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


def each_process_trainer(gpu, args, pos_classes_to_process, all_class_features_meta):
    global_rank = sum(args.gpus[:args.local_rank]) + gpu
    saver_process = 0
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = args.dist_url
        os.environ['MASTER_PORT'] = '9451'
        rpc.init_rpc(f"{global_rank}", rank=global_rank, world_size=args.world_size)

    if global_rank==saver_process:
        model_saver.initializer(args)

    torch.cuda.set_device(gpu)

    positive_features_generator = data_prep.read_features(args, pos_classes_to_process)
    to_write=[]
    for pos_cls_name, positive_cls_feature in positive_features_generator:
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
                start = all_class_features_meta['start_indxs'][batch_no][pos_cls_name]
                ### store the positive distances and use only negatives for tail
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
            sortedTensor = torch.topk(distances,
                                      min(args.tailsize,distances.shape[1]),
                                      dim = 1,
                                      largest = False,
                                      sorted = True).values
            del distances
            negative_distances.append(sortedTensor.cpu())
        sortedTensor = negative_distances[0].to(f"cuda:{gpu}")
        for n in negative_distances[1:]:
            distance_batch = torch.cat([sortedTensor,n.to(f"cuda:{gpu}")], dim=1)
            sortedTensor = torch.topk(distance_batch,
                                      args.tailsize,
                                      dim=1,
                                      largest=False,
                                      sorted=True).values
        del distance_batch
        del negative_distances
        del neg_features
        del positive_cls_feature
        weibull_model = fit_weibull(sortedTensor,
                                    args.distance_multiplier, args.tailsize, gpu)
        del sortedTensor
        extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                     positive_distances.to(f"cuda:{gpu}"),
                                                                                     args.cover_threshold)
        to_write.append((pos_cls_name,
                         extreme_vectors_models,
                         extreme_vectors_indexes,
                         covered_vectors))
        if len(to_write) >= 10 and len(to_write) % 10 == gpu:
            # WARNING: remote can make debugging hell :P
            _ = rpc.remote(f"{saver_process}",
                           model_saver.save_multi_cls_evs,
                           args=(to_write,))
            to_write = []

        del weibull_model

    if args.world_size > 1:
        if len(to_write) > 0:
            _ = rpc.remote(f"{saver_process}",
                           model_saver.save_multi_cls_evs,
                           args=(to_write,))
        rpc.shutdown()

    if global_rank==saver_process:
        model_saver.close()


def trainer(args):
    all_class_features_meta, class_names = data_prep.prep_all_features(args)
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

    if args.world_size > 1:
        for p in processes:
            p.join()
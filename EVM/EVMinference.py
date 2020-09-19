import math
import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import h5py
import EVM
from EVM import data_prep
from EVM import pairwisedistances
from EVM import test_saver
from DistributionModels import weibull

def each_process_inferencer(gpu, test_file_name, args, combination_dict, classes_to_process, all_class_evs):
    global_rank = sum(args.gpus[:args.local_rank]) + gpu
    saver_process = args.world_size
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = args.dist_url
        os.environ['MASTER_PORT'] = '9451'
        print(f"Starting Process No {global_rank}/{args.world_size+1}")
        rpc.init_rpc(f"{global_rank}", rank=global_rank, world_size=args.world_size+1,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=args.world_size*3,
                                                                           rpc_timeout=0)
                     )

    if global_rank==saver_process:
        print(f"Started Saver Process {global_rank}/{args.world_size+1}")
        test_saver.initializer(args, test_file_name = test_file_name, combination_dict=combination_dict)
        while True:
            if test_saver.cls_counter()>=args.total_no_of_classes:
                break
        test_saver.close()
        print(f"Shutting down RPC for Process No {global_rank}/{args.world_size+1}")
        rpc.shutdown()
        return

    torch.cuda.set_device(gpu)

    test_features_generator = data_prep.read_features(args,
                                                      feature_file_names=(test_file_name,),
                                                      cls_to_process=classes_to_process)
    for test_cls_name, test_cls_feature in test_features_generator:
        test_cls_feature = test_cls_feature.to(f"cuda:{gpu}")
        results = {}
        for batch_no, ev_data in enumerate(all_class_evs):
            norm = ev_data['norm_t'].to(f"cuda:{gpu}")
            # distances is a tensor of size no_of_samples_in_test_class X no_of_extreme_vector
            distances = pairwisedistances.cosine_distance(test_cls_feature,
                                                          ev_data['features_t'].to(f"cuda:{gpu}"),
                                                          w2_t=norm)
            del norm
            mr = weibull.weibull(ev_data['weibulls'])
            probs = mr.wscore(distances)
            del distances
            for ev_cls_name in sorted(list(ev_data['start_indx'].keys())):
                start, end = ev_data['start_indx'][ev_cls_name]
                results[ev_cls_name] = torch.max(probs[:, start: end], dim=1).values.cpu()
            del probs
        del test_cls_feature
        cls_results=[]
        for k in sorted(list(results.keys())):
            cls_results.append(results[k][:, None])
        cls_results = torch.cat(cls_results,dim=1)

        # Send the computed information for the current class to the saver process
        _ = rpc.remote(f"{saver_process}",
                       test_saver.save_cls_results,
                       timeout=0,
                       args=(test_cls_name,
                             cls_results))

    if args.world_size > 1:
        print(f"Shutting down RPC for Process No {global_rank}/{args.world_size+1}")
        rpc.shutdown()



def main(args):
    all_evm_param_combinations, _ = EVM.get_all_evm_combinations(args)
    for evm_param_combination in all_evm_param_combinations:
        ev_batches = model_saver.model_loader(args, evm_param_combination)
        # Process each test file
        for test_feature_file in args.test_feature_files:
            # Split Classes according to available gpus
            with h5py.File(test_feature_file, "r") as hf:
                class_names = sorted(list(hf.keys()))
            # TODO: FIX Feature file and remove this
            if '360' in test_feature_file:
                class_names = list(set(class_names)-set(['n01440764']))
            args.total_no_of_classes = len(class_names)

            no_of_classes_each_gpu = math.ceil(args.total_no_of_classes/(args.world_size))
            classes_to_process_on_each_gpu = []
            for indx in range(0,args.total_no_of_classes,no_of_classes_each_gpu):
                classes_to_process_on_each_gpu.append(class_names[indx:indx+no_of_classes_each_gpu])

            if args.world_size == 1:
                each_process_trainer(0, args, classes_to_process_on_each_gpu[0], all_class_features_meta)
            else:
                processes = []
                no_of_processes = args.gpus[args.local_rank]
                if args.local_rank==len(args.gpus)-1:
                    saver_process = mp.Process(target=each_process_inferencer,
                                               args=(args.world_size,
                                                     test_feature_file,
                                                     args,
                                                     evm_param_combination._asdict(),
                                                     None,
                                                     None))
                    saver_process.start()
                    processes.append(saver_process)
                for rank in range(no_of_processes):
                    p = mp.Process(target=each_process_inferencer,
                                   args=(rank,
                                         test_feature_file,
                                         args,
                                         evm_param_combination._asdict(),
                                         classes_to_process_on_each_gpu[rank],
                                         ev_batches))
                    p.start()
                    processes.append(p)

            if args.world_size > 1:
                for p in processes:
                    p.join()
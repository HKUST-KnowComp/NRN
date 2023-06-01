import argparse
from gqe import GQE
from q2b import Q2B
from q2p import Q2P


import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator, separate_query_dict
from dataloader import baseline_abstraction, abstraction
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import json
import networkx as nx

from numeral_encoder import PositionalEncoder, DICE, GMM_Prototype, DigitRNN
import time

torch.autograd.set_detect_anomaly(True)

def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work. 
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True
            
        
        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument('--query_data_dir', default="sampled_data_small", help="The path to the sampled queries.")
    parser.add_argument('--kg_data_dir', default="KG_data/", help="The path the original kg data")

    parser.add_argument("--train_query_dir", required=True)
    parser.add_argument("--valid_query_dir", required=True)
    parser.add_argument("--test_query_dir", required=True)

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.1, type=float)
    parser.add_argument('-nls', "--numerical_label_smoothing", default=0.3, type=float)

    parser.add_argument('--max_train_step', default=370000, type=int)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

   
    parser.add_argument("--experiment_number", type=str, default="e34")

    parser.add_argument("--numeral_encoder", default="dice", type=str)
    parser.add_argument("--mixed_value_reprerentation", action="store_true")

    parser.add_argument("--small", action="store_true")

    parser.add_argument("--timing", action="store_true")

    parser.add_argument("--typed", action="store_true")

    parser.add_argument("--quantile", action="store_true")

    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)


    args = parser.parse_args()


    data_name = args.data_name

    train_data_prefix = data_name + "_train_queries"
    valid_data_prefix = data_name + "_valid_queries"
    test_data_prefix = data_name + "_test_queries"


    if "json" in args.train_query_dir:
        with open(args.train_query_dir, "r") as fin:
            train_data_dict = json.load(fin)
    else:
        with open(args.train_query_dir, "rb") as fin:
            train_data_dict = pickle.load(fin)

    if "json" in args.valid_query_dir:
        with open(args.valid_query_dir, "r") as fin:
            valid_data_dict = json.load(fin)
    else:
        with open(args.valid_query_dir, "rb") as fin:
            valid_data_dict = pickle.load(fin)

    if "json" in args.test_query_dir:
        with open(args.test_query_dir, "r") as fin:
            test_data_dict = json.load(fin)
    else:
        with open(args.test_query_dir, "rb") as fin:
            test_data_dict = pickle.load(fin)


    data_dir =  args.data_name

    if args.small:
        print("Load Train Graph " + data_dir)
        train_path = "../preprocessing/" + data_dir + "_small_train_with_units.pkl"
        train_graph = nx.read_gpickle(train_path)

        print("Load Test Graph " + data_dir)
        test_path = "../preprocessing/" + data_dir + "_small_test_with_units.pkl"
        test_graph = nx.read_gpickle(test_path)
    
    else:
        print("Load Train Graph " + data_dir)
        train_path = "../preprocessing/" + data_dir + "_train_with_units.pkl"
        train_graph = nx.read_gpickle(train_path)

        print("Load Test Graph " + data_dir)
        test_path = "../preprocessing/" + data_dir + "_test_with_units.pkl"
        test_graph = nx.read_gpickle(test_path)


    all_values = []
    for u in test_graph.nodes():
        if isinstance(u, tuple):
            all_values.append(u[0])

    train_values = []
    for u in train_graph.nodes():
        if isinstance(u, tuple):
            train_values.append(u[0])


    all_typed_values = {}
    for u in test_graph.nodes():
        if isinstance(u, tuple):
            if u[1] not in all_typed_values:
                all_typed_values[u[1]] = []
            all_typed_values[u[1]].append(u[0])
        


    EncoderType = PositionalEncoder

    if args.numeral_encoder == "positional":
        EncoderType = PositionalEncoder
    
    elif args.numeral_encoder == "dice":
        EncoderType = DICE

    elif args.numeral_encoder == "gmm":
        EncoderType = GMM_Prototype
    
    else:
        raise ValueError("Invalid numeral encoder")


    train_typed_values = {}
    for u in train_graph.nodes():
        if isinstance(u, tuple):
            if u[1] not in train_typed_values:
                train_typed_values[u[1]] = []
            train_typed_values[u[1]].append(u[0])


    if args.typed:
        encoder_list  = []

        if args.quantile:
            for i in range(len(all_typed_values)):
                positional_encoder_log = EncoderType(output_size=300, train_values=train_typed_values[i], all_values=all_typed_values[i], scaler="quantile")
                encoder_list.append(positional_encoder_log)
        

        else:

            for i in range(len(all_typed_values)):
                positional_encoder_log = EncoderType(output_size=300, train_values=train_typed_values[i], all_values=all_typed_values[i])
                encoder_list.append(positional_encoder_log)
        
    else:
        encoder = EncoderType(output_size=300, train_values=train_values, all_values=all_values)
        encoder_list = [encoder]



    entity_counter = 0
    value_counter = 0
    tuple_all_values = []

    for u in test_graph.nodes():
        if isinstance(u, tuple):
            value_counter += 1
            tuple_all_values.append(u)
        elif isinstance(u, str):
            entity_counter += 1

    value_vocab = dict(zip(tuple_all_values, range(0, len(tuple_all_values))))


    relation_edges_list = []
    attribute_edges_list = []
    reverse_attribute_edges_list = []
    numerical_edges_list = []


    for u, v, a in test_graph.edges(data=True):
        if isinstance(u, tuple) and isinstance(v, tuple):
            for key, value in a.items():
                numerical_edges_list.append(key)
        elif isinstance(u, tuple):
            for key, value in a.items():
                reverse_attribute_edges_list.append(key)
        elif isinstance(v, tuple):
            for key, value in a.items():
                attribute_edges_list.append(key)
        elif isinstance(u, str) and isinstance(v, str):
            for key, value in a.items():
                relation_edges_list.append(key)

    relation_edges_list = list(set(relation_edges_list))
    attribute_edges_list = list(set(attribute_edges_list))
    reverse_attribute_edges_list = list(set(reverse_attribute_edges_list))
    numerical_edges_list = list(set(numerical_edges_list))

    nentity = entity_counter
    nvalue = value_counter

    if args.small:
        nrelation = max(relation_edges_list) + 10
        nattribute = max(attribute_edges_list)+ 10

        nnumerical_proj = max(numerical_edges_list)+ 10
    
    else:

        nrelation = len(relation_edges_list)
        nattribute = len(attribute_edges_list)

        nnumerical_proj = len(numerical_edges_list)


    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '../logs/gradient_tape/' + current_time + "_" + args.model + "_" + data_name + "_" + args.numeral_encoder  
    test_log_dir = '../logs/gradient_tape/' + current_time + "_" + args.model + "_" + data_name + "_" + args.numeral_encoder 
    
    if args.small:
        train_log_dir = train_log_dir + "_small"
        test_log_dir = test_log_dir + "_small"

    if args.typed:
        train_log_dir = train_log_dir + "_typed"
        test_log_dir = test_log_dir + "_typed"
    
    if args.quantile:
        train_log_dir = train_log_dir + "_quantile"
        test_log_dir = test_log_dir + "_quantile"

    train_log_dir += "/train"
    test_log_dir += "/test"
    

    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    batch_size = args.batch_size

    # Test train iterators
    extended_train_data_dict = {}
    extended_train_query_types = []
    extended_train_query_types_counts = []
    extended_train_query_iterators = {}

    for query_type, query_answer_dict in train_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)
        print("Number of queries of type " + query_type + ": " + str(len(sub_query_types_dicts)))

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            extended_train_query_types.append(sub_query_type)
            extended_train_query_types_counts.append(len(sub_query_types_dict))
            extended_train_data_dict[sub_query_type] = sub_query_types_dict
    
    extended_train_query_types_counts = np.array(extended_train_query_types_counts) / np.sum(extended_train_query_types_counts)
    # print(extended_train_query_types_counts)
    # print(np.sum(extended_train_query_types_counts))
    print("Extended query types: ", len(extended_train_query_types))

    # Add training iterators
    for query_type in extended_train_query_types:
        query_answer_dict = extended_train_data_dict[query_type]
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict,
                         baseline=False, nattribute=nattribute, value_vocab=value_vocab),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        extended_train_query_iterators[query_type] = new_iterator





    print("====== Create Development Dataloader ======")
    baseline_validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, sub_query_types_dict,
                             baseline=False, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            baseline_validation_loaders[sub_query_type] = new_iterator


    print("====== Create Testing Dataloader ======")
    baseline_test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                TestDataset(nentity, nrelation, sub_query_types_dict,
                            baseline=False, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )
            baseline_test_loaders[sub_query_type] = new_iterator

    if args.model == "q2p":
        # model = Q2P(num_entities=nentity + nvalue,
        #             num_relations=nrelation + nattribute*2 + nnumerical_proj,
        #             embedding_size=300)

        model = Q2P(num_entities=nentity,
                    num_relations=nrelation,
                    embedding_size=300,
                    num_attributes=nattribute,
                    num_numrical_proj=nnumerical_proj,
                    value_vocab=value_vocab,
                    number_encoder_list=encoder_list,
                    mixed_value_reprerentation=True,
                    label_smoothing=args.label_smoothing,
                    numerical_label_smoothing=args.numerical_label_smoothing,)
    elif args.model == "gqe":

        model = GQE(num_entities=nentity,
                    num_relations=nrelation,
                    embedding_size=300,
                    num_attributes=nattribute,
                    num_numrical_proj=nnumerical_proj,
                    value_vocab=value_vocab,
                    number_encoder_list=encoder_list,
                    mixed_value_reprerentation=True,
                    label_smoothing=args.label_smoothing,
                    numerical_label_smoothing=args.numerical_label_smoothing,)
                    
    
    elif args.model == "q2b":

        model = Q2B(num_entities=nentity,
                    num_relations=nrelation,
                    embedding_size=300,
                    num_attributes=nattribute,
                    num_numrical_proj=nnumerical_proj,
                    value_vocab=value_vocab,
                    number_encoder_list=encoder_list,
                    mixed_value_reprerentation=True,
                    label_smoothing=args.label_smoothing,
                    numerical_label_smoothing=args.numerical_label_smoothing,)
    else:
        raise NotImplementedError


    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    print("====== Training ======")

    if args.timing:
        inference_time = []
        training_time = []

    for step in tqdm(range(args.max_train_step )):
        model.train()
        optimizer.zero_grad()

        task_name = np.random.choice(extended_train_query_types, p=extended_train_query_types_counts)
        iterator = extended_train_query_iterators[task_name]
        batched_query, unified_ids, positive_sample = next(iterator)

        if args.model == "lstm" or args.model == "transformer":
            batched_query = unified_ids
        
        start_time = time.time()

        loss = model(batched_query, positive_sample)

        if args.timing:
            inference_time.append(time.time() - start_time)

        if args.gradient_accumulation_steps > 1:
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        else: 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if args.timing:
            training_time.append(time.time() - start_time)
        # loss.backward()
        # optimizer.step()

        if args.timing and step == 100:
            with open("../logs/" + model_name + "_" + data_name + "_time.txt", "w") as fout:
                fout.write("Inference Time: " + str(np.mean(inference_time)))
                fout.write("Training Time: " + str(np.mean(training_time)))
            
            exit()


        if step % 100 == 0:
            train_summary_writer.add_scalar("y-train-" + task_name, loss.item(), step)

        save_step = args.log_steps
        model_name = args.model

        if step % save_step == 0 and not args.timing:
            general_checkpoint_path = "../logs/" + model_name + "_" + str(step) + "_" + data_name + "_" + args.numeral_encoder
            if args.typed:
                general_checkpoint_path += "_typed"
            if args.small:
                general_checkpoint_path += "_small"
            if args.quantile:
                general_checkpoint_path += "_quantile"

            general_checkpoint_path += ".bin"

            if torch.cuda.device_count() > 1:
                torch.save({
                    'steps': step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            else:
                torch.save({
                    'steps': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)

        early_stop = False
        # conduct testing on validation and testing
        if step % save_step == 0 and not args.timing:
            model.eval()

            
            all_implicit_generalization_logs = []

            
            abstracted_implicit_generalization_logs = {}
            

            for task_name, loader in baseline_validation_loaders.items():

                
                all_generalization_logs = []

                

                for batched_query, unified_ids, train_answers, valid_answers in tqdm(loader):

                    if args.model == "lstm" or args.model == "transformer":
                        batched_query = unified_ids

                    query_embedding = model(batched_query)
                   
                    generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

                   
                    all_generalization_logs.extend(generalization_logs)

                    if loader.dataset.isImplicit:
                        
                        all_implicit_generalization_logs.extend(generalization_logs)

                        abstract_query_type = loader.dataset.query_type_abstract
                        
                        
                        if abstract_query_type in abstracted_implicit_generalization_logs:
                            abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                        else:
                            abstracted_implicit_generalization_logs[abstract_query_type] = []
                            abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                        
                    
                  

                    if early_stop:
                        break

                aggregated_generalization_logs = log_aggregation(all_generalization_logs)

                


                for key, value in aggregated_generalization_logs.items():
                    test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, step)

     

            
            aggregated_implicit_generalization_logs = log_aggregation(all_implicit_generalization_logs)
            

           

            for key, value in aggregated_implicit_generalization_logs.items():
                test_summary_writer.add_scalar("x-valid-implicit-" + key, value, step)

          
            
            for key, value in abstracted_implicit_generalization_logs.items():
                aggregated_value = log_aggregation(value)
                for metric, metric_value in aggregated_value.items():
                    test_summary_writer.add_scalar("y-valid-implicit-" + key + "-" + metric, metric_value, step)


            all_implicit_generalization_logs = []

          
            abstracted_implicit_generalization_logs = {}

            for task_name, loader in baseline_test_loaders.items():

                all_generalization_logs = []

                for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
                    if args.model == "lstm" or args.model == "transformer":
                        batched_query = unified_ids
                    
                    query_embedding = model(batched_query)
                    
                    generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                    
                    all_generalization_logs.extend(generalization_logs)

                    if loader.dataset.isImplicit:
                        
                        all_implicit_generalization_logs.extend(generalization_logs)

                        abstract_query_type = loader.dataset.query_type_abstract
                       
                        
                        if abstract_query_type in abstracted_implicit_generalization_logs:
                            abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                        else:
                            abstracted_implicit_generalization_logs[abstract_query_type] = []
                            abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                    
  
                       
                    if early_stop:
                        break

                aggregated_generalization_logs = log_aggregation(all_generalization_logs)

                

                for key, value in aggregated_generalization_logs.items():
                    test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, step)

        
            aggregated_implicit_generalization_logs = log_aggregation(all_implicit_generalization_logs)


            for key, value in aggregated_implicit_generalization_logs.items():
                test_summary_writer.add_scalar("x-test-implicit-" + key, value, step)

         
                
            for key, value in abstracted_implicit_generalization_logs.items():
                aggregated_value = log_aggregation(value)
                for metric, metric_value in aggregated_value.items():
                    test_summary_writer.add_scalar("y-test-implicit-" + key + "-" + metric, metric_value, step)
        

            gc.collect()




















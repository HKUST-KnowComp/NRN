import json
import pickle
from tqdm import  tqdm
import os


dir_list = ["../sampled_data"]
output_dir = "../input_files/"

for directory_name in dir_list:

    data_names = ["FB15K", "DB15K", "YAGO15K"]

    all_files = os.listdir(directory_name)
    sample_data_path = directory_name + "/"

    def merge_query_file(query_file_dict_list):
        """
        The query file list is a list of dictionary of the train/validation/test queries that are separately sampled
        """
        merged_dict = {}

        for query_file_dict in query_file_dict_list:
            for query_type in query_file_dict.keys():
                if query_type in merged_dict:
                    for query, answer_dict in query_file_dict[query_type].items():
                        merged_dict[query_type][query] = answer_dict
                else:
                    merged_dict[query_type] = {}
                    for query, answer_dict in query_file_dict[query_type].items():
                        merged_dict[query_type][query] = answer_dict

        print({k: len(v) for k, v in merged_dict.items()})

        return merged_dict

    

    for data_name in data_names:
        print(data_name)

        train_data_prefix = data_name  + "_train_queries"
        valid_data_prefix = data_name  + "_valid_queries"
        test_data_prefix = data_name  + "_test_queries"

        train_dict_list_same = []
        # print("train")
        for file in tqdm(all_files):
            if train_data_prefix in file:

                with open(sample_data_path + file, "r") as fin:
                    data_dict = json.load(fin)
                    train_dict_list_same.append(data_dict)

        # print("#same: ", len(train_dict_list_same))
        train_data_dict_same = merge_query_file(train_dict_list_same)

        filehandler = open(output_dir + train_data_prefix  + ".pkl", "wb")
        pickle.dump(train_data_dict_same, filehandler)
        filehandler.close()

        # print("valid")
        valid_dict_list = []
        for file in tqdm(all_files):
            if valid_data_prefix in file:
                with open(sample_data_path + file, "r") as fin:
                    data_dict = json.load(fin)
                    valid_dict_list.append(data_dict)
        if len(valid_dict_list) > 0:
            valid_data_dict = merge_query_file(valid_dict_list)

            filehandler = open(output_dir + valid_data_prefix  + ".pkl", "wb")
            pickle.dump(valid_data_dict, filehandler)
            filehandler.close()

        # print("test")
        test_dict_list = []
        for file in tqdm(all_files):
            if test_data_prefix in file:
                with open(sample_data_path + file, "r") as fin:
                    data_dict = json.load(fin)
                    test_dict_list.append(data_dict)
        if len(test_dict_list) > 0:
            test_data_dict = merge_query_file(test_dict_list)

            filehandler = open(output_dir + test_data_prefix + ".pkl", "wb")
            pickle.dump(test_data_dict, filehandler)
            filehandler.close()


#!/usr/bin/python3

import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import pickle

# Add unified tokens, expressing both relations, entities, and logical operations

special_token_dict = {
    "(": 0,
    ")": 1,
    "p": 2,
    "i": 3,
    "u": 4,
    "n": 5,
    "rp": 6,
    "ap": 7,
    "rap": 8,
    "np": 9
}
std = special_token_dict
std_offset = 100


def baseline_abstraction(instantiated_query, nentity, nrelation, nattribute, value_vocabulary):
    query = instantiated_query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    def r(relation_id):
        return relation_id + nentity + std_offset

    def e(entity_id):
        return entity_id + std_offset

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    if sub_queries[0] == "p":

        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[2], nentity, nrelation, nattribute, value_vocabulary)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "ap":

        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[2], nentity, nrelation, nattribute, value_vocabulary)
        relation_id = int(sub_queries[1][1:-1]) + nrelation
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "rp":

        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[2], nentity, nrelation, nattribute, value_vocabulary)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "rap":

        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[2], nentity, nrelation, nattribute, value_vocabulary)
        relation_id = int(sub_queries[1][1:-1]) + nrelation
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "np":

        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[2], nentity, nrelation, nattribute, value_vocabulary)
        relation_id = int(sub_queries[1][1:-1]) + nrelation + 2 * nattribute
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "e":

        entity_id = int(sub_queries[1][1:-1])

        ids_list = [entity_id]
        this_query_type = "(e)"
        this_unified_ids = [std["("], e(entity_id), std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "nv":

        relation_type_id, value  = sub_queries[1][1:-1].split(",")
        value = float(value)
        relation_type_id = int(relation_type_id) 


        value_tuple = (value, relation_type_id)


        value_id = value_vocabulary[value_tuple] + nentity

        ids_list = [value_id]
        this_query_type = "(e)"
        this_unified_ids = [std["("], value_id, std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "i":

        ids_list = []
        this_query_type = "(i"
        this_unified_ids = [std["("], std["i"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[i], nentity, nrelation, nattribute, value_vocabulary)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "u":
        ids_list = []
        this_query_type = "(u"
        this_unified_ids = [std["("], std["u"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[i], nentity, nrelation, nattribute, value_vocabulary)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "n":
        sub_ids_list, sub_query_type, sub_unified_ids = baseline_abstraction(sub_queries[1], nentity, nrelation, nattribute, value_vocabulary)
        return sub_ids_list, "(n," + sub_query_type + ")", [std["("], std["n"]] + sub_unified_ids + [std[")"]]

    else:
        print("Invalid Pattern")
        exit()



#  Add multiple types for abstraction
def abstraction(instantiated_query, nentity, nrelation):
    query = instantiated_query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    def r(relation_id):
        return relation_id + nentity + std_offset

    def e(entity_id):
        return entity_id + std_offset

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    if sub_queries[0] == "p":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "ap":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(ap," + sub_query_type + ")"
        this_unified_ids = [std["("], std["ap"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "rp":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(rp," + sub_query_type + ")"
        this_unified_ids = [std["("], std["rp"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "rap":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(rap," + sub_query_type + ")"
        this_unified_ids = [std["("], std["rap"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "np":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation)
        relation_id = int(sub_queries[1][1:-1])
        ids_list = [relation_id] + sub_ids_list
        this_query_type = "(np," + sub_query_type + ")"
        this_unified_ids = [std["("], std["np"], r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "e":

        entity_id = int(sub_queries[1][1:-1])

        ids_list = [entity_id]
        this_query_type = "(e)"
        this_unified_ids = [std["("], e(entity_id), std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "nv":


        relation_type_id, value  = sub_queries[1][1:-1].split(",")
        value = float(value)
        relation_type_id = int(relation_type_id) 

        

        ids_list = [relation_type_id, value]
        this_query_type = "(nv)"
        this_unified_ids = [std["("], value, std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "i":

        ids_list = []
        this_query_type = "(i"
        this_unified_ids = [std["("], std["i"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "u":
        ids_list = []
        this_query_type = "(u"
        this_unified_ids = [std["("], std["u"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "n":
        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[1], nentity, nrelation)
        return sub_ids_list, "(n," + sub_query_type + ")", [std["("], std["n"]] + sub_unified_ids + [std[")"]]

    else:
        print("Invalid Pattern")
        exit()

#  add multiple types for Instantiation
class Instantiation(object):

    def __init__(self, value_matrix):
        self.value_matrix = np.array(value_matrix, dtype=float)

    def instantiate(self, query_pattern):

        query = query_pattern[1:-1]
        parenthesis_count = 0

        sub_queries = []
        jj = 0

        for ii, character in enumerate(query):
            # Skip the comma inside a parenthesis
            if character == "(":
                parenthesis_count += 1

            elif character == ")":
                parenthesis_count -= 1

            if parenthesis_count > 0:
                continue

            if character == ",":
                sub_queries.append(query[jj: ii])
                jj = ii + 1

        sub_queries.append(query[jj: len(query)])

        if sub_queries[0] == "p":

            relation_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return "p", relation_ids, sub_batched_query

        elif sub_queries[0] == "ap":

            relation_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return "ap", relation_ids, sub_batched_query

        elif sub_queries[0] == "rap":

            relation_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return "rap", relation_ids, sub_batched_query

        elif sub_queries[0] == "rp":

            relation_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return "rp", relation_ids, sub_batched_query

        elif sub_queries[0] == "np":

            relation_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return "np", relation_ids, sub_batched_query

        elif sub_queries[0] == "e":
            entity_ids = self.value_matrix[:, 0].astype(int)
            self.value_matrix = self.value_matrix[:, 1:]

            return "e", entity_ids

        elif sub_queries[0] == "nv":
            values_types_ids = self.value_matrix[:, 0].astype(int)
            numerical_values = self.value_matrix[:, 1].astype(float)
            self.value_matrix = self.value_matrix[:, 2:]

            return "nv", values_types_ids, numerical_values

        elif sub_queries[0] == "i":

            return_list = ["i"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)

        elif sub_queries[0] == "u":
            return_list = ["u"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)

        elif sub_queries[0] == "n":
            sub_batched_query = self.instantiate(sub_queries[1])

            return "n", sub_batched_query

        else:
            print("Invalid Pattern")
            exit()


# implement this method, where this method is able to further split the queries into more detailed types
def separate_query_dict(query_answers_dict, num_entities, num_relation_types):

    detailed_query_types = {}

    for query_string, answers_dict in query_answers_dict.items():
        ids_list, detailed_query_type, unified_ids = abstraction(query_string, num_entities, num_relation_types)
        if detailed_query_type in detailed_query_types:
            detailed_query_types[detailed_query_type][query_string] = answers_dict
        else:
            detailed_query_types[detailed_query_type] = {query_string: answers_dict}

    return detailed_query_types


class TestDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict, baseline=False, value_vocab=None, nattribute=0):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []

        self.unified_id_list = []

        self.query_type = None
        self.query_type_abstract = None

        for query, answer_list in query_answers_dict.items():
            if baseline:
                this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
            else:

                if self.query_type_abstract is None:
                    this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
                    self.query_type_abstract = this_query_type

                this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)
           
            single_query_answer_list = []
            for ans in answer_list["train_answers"]:
                if isinstance(ans, str):
                    single_query_answer_list.append(int(ans))
                    self.isImplicit = True
                elif baseline:
                    self.isImplicit = False
                    
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])] + nentity) 
                else:
                    self.isImplicit = False
                    
                    single_query_answer_list.append((ans[0], ans[1]))

            self.train_answer_list.append(single_query_answer_list)

            single_query_answer_list = []
            for ans in answer_list["valid_answers"]:
                if isinstance(ans, str):
                    single_query_answer_list.append(int(ans))
                elif baseline:
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])] + nentity)
                else:
                    single_query_answer_list.append((ans[0], ans[1]))

            self.valid_answer_list.append(single_query_answer_list)

            single_query_answer_list = []
            for ans in answer_list["test_answers"]:
                if isinstance(ans, str):
                    single_query_answer_list.append(int(ans))
                elif baseline:
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])] + nentity)
                else:
                    single_query_answer_list.append((ans[0], ans[1]))

            self.test_answer_list.append(single_query_answer_list)

            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, test_answer_list, self.query_type

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]
        test_answers = [_[4] for _ in data]

        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[5] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers, test_answers


class ValidDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict,baseline=False, value_vocab=None, nattribute=0):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.unified_id_list = []

        self.query_type = None
        self.query_type_abstract = None

        for query, answer_list in query_answers_dict.items():
            if baseline:
                this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
            else:
                if self.query_type_abstract is None:
                    this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
                    self.query_type_abstract = this_query_type

                this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)
            
            single_query_answer_list = []
            for ans in answer_list["train_answers"]:
                if isinstance(ans, str):
                    self.isImplicit = True
                    single_query_answer_list.append(int(ans))
                elif baseline:
                    self.isImplicit = False
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])]+ nentity)
                else:
                    self.isImplicit = False
                    single_query_answer_list.append((ans[0], ans[1]))

            self.train_answer_list.append(single_query_answer_list)

            single_query_answer_list = []
            for ans in answer_list["valid_answers"]:
                if isinstance(ans, str):
                    single_query_answer_list.append(int(ans))
                elif baseline:
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])]+ nentity)
                else:
                    single_query_answer_list.append((ans[0], ans[1]))

            self.valid_answer_list.append(single_query_answer_list)


            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)
            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, self.query_type

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]

        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[4] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers


# The design of the data loader is tricky. There are two requirements of loading. 1. We cannot do parsing of each sample
# during the collate time, or the training speed will be too slow. This is because we may use large batch size.
# 2. The data loader must be flexible enough to deal with all types of queries automatically. As for one data loader
# only deal with one type of query, we can store all the numerical values in a separate matrix, and memorize all their
# index. By doing this, we can fast reconstruct the structured and batched result quickly.


class TrainDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict,baseline=False, value_vocab=None, nattribute=0):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.id_list = []
        self.answer_list = []
        self.query_type = None
        self.query_type_abstract = None

        self.unified_id_list = []


        for query, answer_list in query_answers_dict.items():
            if baseline:
                this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
            else:

                if self.query_type_abstract is None:
                    this_id_list, this_query_type, unified_ids = baseline_abstraction(query, nentity, nrelation, nattribute, value_vocab)
                    self.query_type_abstract = this_query_type

                this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation)

            single_query_answer_list = []
            for ans in answer_list["train_answers"]:
                if isinstance(ans, str):
                    single_query_answer_list.append(int(ans))
                    self.isImplicit = True
                elif baseline:
                    self.isImplicit = False

                    
                    single_query_answer_list.append(value_vocab[(ans[0], ans[1])]+ nentity)
                else:
                    self.isImplicit = False
                    single_query_answer_list.append((ans[0], ans[1]))

            self.answer_list.append(single_query_answer_list)
            self.id_list.append(this_id_list)

            self.unified_id_list.append(unified_ids)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        ids_in_query = self.id_list[idx]
        answer_list = self.answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        if isinstance(answer_list[0], int):

            tail = np.random.choice(answer_list)
        else:
            # tail = np.random.choice(answer_list, size=300)
            
            tail_id = np.random.choice(range(len(answer_list)))
            tail = answer_list[tail_id]
        positive_sample = tail

        return ids_in_query, unified_id_list, positive_sample, self.query_type

    @staticmethod
    def collate_fn(data):
        positive_sample = [_[2] for _ in data]
        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[3] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix).instantiate(query_type[0])

        return batched_query, unified_ids, positive_sample


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


def old_main():
    train_data_path = "./FB15k-237-betae_train_queries.json"
    valid_data_path = "./FB15k-237-betae_valid_queries.json"
    test_data_path = "./FB15k-237-betae_test_queries.json"
    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_path = "./KG_data/FB15k-237-betae"

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        batched_query, unified_ids, positive_sample = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = DataLoader(
            ValidDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            TestDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TestDataset.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break


def main():
    train_data_path = "../input_files_small/FB15K_small_train_queries.pkl"
    valid_data_path = "../input_files_small/FB15K_small_valid_queries.pkl"
    test_data_path = "../input_files_small/FB15K_small_test_queries.pkl"
    with open(train_data_path, "rb") as fin:
        train_data_dict = pickle.load(fin)

    with open(valid_data_path, "rb") as fin:
        valid_data_dict = pickle.load(fin)

    with open(test_data_path, "rb") as fin:
        test_data_dict = pickle.load(fin)

    data_dir = "FB15K"
   
    print("Load Train Graph " + data_dir)
    train_path = "../preprocessing/" + data_dir + "_small_train_with_units.pkl"
    train_graph = nx.read_gpickle(train_path)

    print("Load Test Graph " + data_dir)
    test_path = "../preprocessing/" + data_dir + "_small_test_with_units.pkl"
    test_graph = nx.read_gpickle(test_path)



    entity_counter = 0
    value_counter = 0
    all_values = []

    for u in test_graph.nodes():
        if isinstance(u, tuple):
            value_counter += 1
            all_values.append(u)
        elif isinstance(u, str):
            entity_counter += 1

    value_vocab = dict(zip(all_values, range(0, len(all_values))))

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

    nrelation = max(relation_edges_list) + 10
    nattribute = max(attribute_edges_list)+ 10

    nnumerical_proj = max(numerical_edges_list)+ 10

    print("nrelation: ", nrelation)
    print("nattribute: ", nattribute)
    print("nnumerical_proj: ", nnumerical_proj)
    
    


    batch_size = 5


    baseline_validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, sub_query_types_dict,
                             baseline=True, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            baseline_validation_loaders[sub_query_type] = new_iterator

    for key, loader in baseline_validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            

    baseline_test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                TestDataset(nentity, nrelation, sub_query_types_dict, 
                            baseline=True, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )
            baseline_test_loaders[sub_query_type] = new_iterator

    for key, loader in baseline_test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            # print(train_answers)
            # print(valid_answers)
            # print(test_answers)

    

    # Test train iterators
    extended_train_data_dict = {}
    extended_train_query_types = []
    extended_train_query_types_counts = []
    extended_train_query_iterators = []

    for query_type, query_answer_dict in train_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            extended_train_query_types.append(sub_query_type)
            extended_train_query_types_counts.append(len(sub_query_types_dict))
            extended_train_data_dict[sub_query_type] = sub_query_types_dict

    
    print("Extended query types: ", len(extended_train_query_types))
    # Add training iterators
    for query_type in extended_train_query_types:
        query_answer_dict = extended_train_data_dict[query_type]
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict, 
                         baseline=True, nattribute=nattribute, value_vocab=value_vocab),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        extended_train_query_iterators.append(new_iterator)

        batched_query, unified_ids, positive_sample = next(new_iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

        # query_embedding = q2p_model(batched_query)
        # print(query_embedding.shape)
        # loss = q2p_model(batched_query, positive_sample)
        # print(loss)

    




    validation_loaders = {}
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
            validation_loaders[sub_query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            # print(unified_ids)
            print([len(_) for _ in train_answers])
            # print([len(_) for _ in valid_answers])

           

            break

    test_loaders = {}
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
            test_loaders[sub_query_type] = new_iterator

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            # print(unified_ids)
            print(train_answers[0][0])
            try:
                print(type(train_answers[0][0][0]))
            except:
                pass
            # print([len(_) for _ in train_answers])
            # print([len(_) for _ in valid_answers])
            # print([len(_) for _ in test_answers])

    # Test train iterators
    extended_train_data_dict = {}
    extended_train_query_types = []
    extended_train_query_types_counts = []
    extended_train_query_iterators = []

    for query_type, query_answer_dict in train_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            extended_train_query_types.append(sub_query_type)
            extended_train_query_types_counts.append(len(sub_query_types_dict))
            extended_train_data_dict[sub_query_type] = sub_query_types_dict

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
        extended_train_query_iterators.append(new_iterator)

        batched_query, unified_ids, positive_sample = next(new_iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

       
         

        



if __name__ == "__main__":
    main()
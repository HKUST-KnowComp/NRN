import networkx as nx

from random import sample, choice, randint, shuffle
import csv

from datetime import datetime as dt
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import json

n_queries_train_dict_same = {
    "FB15K": 300000,
    "DB15K": 200000,
    "YAGO15K": 200000
}

n_queries_valid_test_dict_same = {
    "FB15K": 30000,
    "DB15K": 20000,
    "YAGO15K": 20000
}

n_queries_train_dict_small = {
    "FB15K": 20,
    "DB15K": 20,
    "YAGO15K": 20
}

n_queries_valid_test_dict_small = {
    "FB15K": 20,
    "DB15K": 20,
    "YAGO15K": 20
}





"""
Here the projections will be extended to four types

rp: relational projection, from entities to entities
ap: attribute projection, from entities to attribute values
rap: reversed attribute projection, from attribute values to entities
np: numerical projections, from attribute values to attribute values

entities are also extended to two types:
nv: numerical values
e: entities


"""




def _pattern_to_query_experiment_34(pattern, graph, node):
    """
    In this function, _pattern_to_query_experiment_34 is recursively used for finding the anchor nodes and relations from
    a randomly sampled entity, which is assumed to be the answer.

    This sampler is different from the normal one. We are going to sample the queries that are used to test the existing
    reasoning model whether they can conduct reasoning over approximately equal quantities. Here

    :param pattern:
    :param graph:
    :param node:
    :return:
    """

    pattern = pattern[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    for ii, character in enumerate(pattern):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(pattern[jj: ii])
            jj = ii + 1

    sub_queries.append(pattern[jj: len(pattern)])

    if sub_queries[0] == "p":
        # if the current node is an entity nodes, we can do relational projection or attribute projection.
        if isinstance(node, str):

            reversely_connected_nodes = [next_node for next_node in list(graph.predecessors(node))]

            # There are some type problem in the predecessor API, we nne
            reversely_connected_value_nodes = [n for n in reversely_connected_nodes if isinstance(n, tuple)]
            reversely_connected_entity_nodes = [n for n in reversely_connected_nodes if isinstance(n, str)]

            # randomly select on of the projection type. Note that this is a reverse search, so the relation name should
            # be either relation projection or reversed attribute projection
            projection_type = choice(["rp", "rap"])

            if projection_type == "rp":
                reverse_target_nodes = reversely_connected_entity_nodes

            else:
                reverse_target_nodes = reversely_connected_value_nodes

            if len(reverse_target_nodes) == 0:
                return None, None

            next_node = choice(reverse_target_nodes)
            edge_name = choice([k for k in graph.edges[next_node, node].keys()])

            sub_query, _ = _pattern_to_query_experiment_34(sub_queries[1], graph, next_node)
            if sub_query is None:
                return None, None

            return "(" + projection_type + ",(" + str(edge_name) + '),' + sub_query + ")", next_node

        elif isinstance(node, tuple):
            # For this experiment, there is no explicit numerical projection

            projection_type = choice(["ap", "np"])

            reversely_connected_nodes = [next_node for next_node in list(graph.predecessors(node))]
            reversely_connected_entity_nodes = [n for n in reversely_connected_nodes if isinstance(n, str)]
            reversely_connected_numerical_nodes = [n for n in reversely_connected_nodes if isinstance(n, tuple)]

            if projection_type == "ap":
                reversely_connected_nodes = reversely_connected_entity_nodes
            else:
                reversely_connected_nodes = reversely_connected_numerical_nodes

            if len(reversely_connected_nodes) == 0:
                return None, None

            next_node = choice(reversely_connected_nodes)

            # min_edge_value = min(list(graph.edges[next_node, node]["total_score"].values()))

            edge_name = choice([k for k in graph.edges[next_node, node].keys()])

            sub_query, _ = _pattern_to_query_experiment_34(sub_queries[1], graph, next_node)
            if sub_query is None:
                return None, None

            return "(" + projection_type + ",(" + str(edge_name) + '),' + sub_query + ")", next_node

        else:
            return None, None

    elif sub_queries[0] == "n":
        """If we use the negation here, it is possible that we generate a query that do not have an answer.
        But the overall chance is small. Anyway, when we cannot find an answer we just sample again.

        After modification, we choose to use the same node for sampling to enable the negation query do have an effect
        on the final outcome
        """

        # random_node = sample(list(graph.nodes()), 1)[0]
        sub_query, returned_node = _pattern_to_query_experiment_34(sub_queries[1], graph, node)
        if sub_query is None:
            return None, None

        return "(n," + sub_query + ")", returned_node

    elif sub_queries[0] == "e":

        if isinstance(node, str):
            return "(e,(" + node + "))", str(node)
        else:
            return "(nv,(" +  str(node[1]) + "," + str(node[0])  + "))", str(node)

    elif sub_queries[0] == "i":

        sub_queries_list = []

        next_node_list = []

        for i in range(1, len(sub_queries)):
            sub_q, _next_node = _pattern_to_query_experiment_34(sub_queries[i], graph, node)
            sub_queries_list.append(sub_q)

            next_node_list.append(_next_node)

        for sub_query in sub_queries_list:
            if sub_query is None:
                return None, None

        for index_i, sub_query_i in enumerate(sub_queries_list):
            for index_j in range(index_i + 1, len(sub_queries_list)):
                if sub_query_i == sub_queries_list[index_j]:
                    return None, None

                if next_node_list[index_i] == next_node_list[index_j]:
                    return None, None

        return_str = "(i"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return return_str, node

    elif sub_queries[0] == "u":
        # randomly sample a node
        sub_queries_list = []
        next_node_list = []

        random_subquery_index = randint(1, len(sub_queries) - 1)

        # The answer only need to be one of the queries
        for i in range(1, len(sub_queries)):
            if i == random_subquery_index:
                sub_q, _next_node = _pattern_to_query_experiment_34(sub_queries[i], graph, node)
            else:
                while True:
                    random_node = sample(list(graph.nodes()) , 1)[0]
                    if type(random_node) == type(node):
                        break
                sub_q, _next_node = _pattern_to_query_experiment_34(sub_queries[i], graph, random_node)

            if sub_q is None:
                return None, None

            sub_queries_list.append(sub_q)
            next_node_list.append(_next_node)

        if len(sub_queries_list) == 0:
            return None, None

        return_str = "(u"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return return_str, node

    else:
        print("Invalid Pattern")
        exit()




class GraphSamplerE34:
    def __init__(self, graph):

        self.graph = graph
        self.dense_nodes = list(graph.nodes)

    # The function used to call the recursion of sampling queries from the ASER graph.
    def sample_with_pattern(self, pattern):
        while True:
            random_node = sample(self.dense_nodes, 1)[0]
            _query, _ = _pattern_to_query_experiment_34(pattern, self.graph, random_node)
            if _query is not None:
                return _query

    def iterative_sample_with_pattern(self, pattern="(p,(e))"):

        result_query_list = []
        for node in tqdm(self.dense_nodes):
            _query, _ = _pattern_to_query_experiment_34(pattern, self.graph, node)
            if _query is not None:
                result_query_list.append(_query)

        return result_query_list

    def generate_one_p_queries(self):
        result_query_list = []
        for node in tqdm(self.dense_nodes):
            for tail_node, attribute_dict in self.graph[node].items():
                # "(p,(40),(e,(2429)))"

                #  Fix the projection type and node type according to the node type and tail_node type

                if isinstance(node, str) and isinstance(tail_node, str):
                    projection_type = "rp"
                    entity_type = "e"

                elif isinstance(node, str) and isinstance(tail_node, tuple):
                    projection_type = "ap"
                    entity_type = "e"

                elif isinstance(node, tuple) and isinstance(tail_node, str):
                    projection_type = "rap"
                    entity_type = "nv"

                else:
                    projection_type = "np"
                    entity_type = "nv"

                if entity_type == "e":
                    for key in attribute_dict.keys():
                        query = "(" + projection_type + ",(" + str(key) + "),(" + entity_type + ",(" + str(node) + ")))"
                        result_query_list.append(query)
                
                else:
                    for key in attribute_dict.keys():
                        query = "(" + projection_type + ",(" + str(key) + "),(" + entity_type + ",(" + str(node[1]) + "," + str(node[0]) + ")))"
                        result_query_list.append(query)

        return list(set(result_query_list))

    # The function used for finding the answers to a query
    def query_search_answer(self, query):

        graph = self.graph

        query = query[1:-1]
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

        if sub_queries[0] == "rp":

            sub_query_answers = self.query_search_answer(sub_queries[2])
            # print(sub_queries[0])
            # print(sub_query_answers)
            all_answers = []

            for answer_id, sub_answer in enumerate(sub_query_answers):

                if not isinstance(sub_answer, str):
                    continue

                try:
                    next_nodes = graph.neighbors(sub_answer)
                except:
                    next_nodes = []

                for node in next_nodes:
                    if not isinstance(node, str):
                        continue
                    if int(sub_queries[1][1:-1]) in graph.edges[sub_answer, node]:
                        all_answers.append(node)
            all_answers = list(set(all_answers))
            return all_answers

        elif sub_queries[0] == "ap":

            sub_query_answers = self.query_search_answer(sub_queries[2])
            # print(sub_queries[0])
            # print(sub_query_answers)
            all_answers = []

            for answer_id, sub_answer in enumerate(sub_query_answers):

                if not isinstance(sub_answer, str):
                    continue

                try:
                    next_nodes = graph.neighbors(sub_answer)
                except:
                    next_nodes = []

                for node in next_nodes:
                    if not isinstance(node, tuple):
                        continue
                    if int(sub_queries[1][1:-1]) in graph.edges[sub_answer, node]:
                        all_answers.append(node)

            all_answers = list(set(all_answers))
            return all_answers

        elif sub_queries[0] == "rap":
            sub_query_answers = self.query_search_answer(sub_queries[2])
            
            all_answers = []

            for answer_id, sub_answer in enumerate(sub_query_answers):

                if not isinstance(sub_answer, tuple):
                    continue

                try:
                    next_nodes = graph.neighbors(sub_answer)
                except:
                    next_nodes = []

                for node in next_nodes:
                    if not isinstance(node, str):
                        continue
                    if int(sub_queries[1][1:-1]) in graph.edges[sub_answer, node]:
                        all_answers.append(node)
            all_answers = list(set(all_answers))
            return all_answers

        elif sub_queries[0] == "np":
            sub_query_answers = self.query_search_answer(sub_queries[2])
           
            all_answers = []

            for answer_id, sub_answer in enumerate(sub_query_answers):

                if not isinstance(sub_answer, tuple):
                    continue

                try:
                    next_nodes = graph.neighbors(sub_answer)
                except:
                    next_nodes = []

                for node in next_nodes:
                    if not isinstance(node, tuple):
                        continue
                    if int(sub_queries[1][1:-1]) in graph.edges[sub_answer, node]:
                        all_answers.append(node)
            all_answers = list(set(all_answers))
            return all_answers

        elif sub_queries[0] == "e":

            return [sub_queries[1][1:-1]]

        elif sub_queries[0] == "nv":
            """
            Numerical values
            """

            # print(sub_queries[0])
            # print(float(sub_queries[1][1:-1]))

            type_id, numerical_value = sub_queries[1][1:-1].split(",")

            return [(float(numerical_value), int(type_id))]

        elif sub_queries[0] == "i":

            sub_query_answers_list = []

            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.query_search_answer(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers & set(sub_query_answers)

            merged_answers = list(merged_answers)

            return merged_answers

        elif sub_queries[0] == "u":

            sub_query_answers_list = []
            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.query_search_answer(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers | set(sub_query_answers)

            merged_answers = list(merged_answers)

            return merged_answers
        elif sub_queries[0] == "n":
            sub_query_answers = self.query_search_answer(sub_queries[1])
            all_nodes = list(self.graph.nodes)
            negative_answers = [node for node in all_nodes if node not in sub_query_answers]

            negative_answers = list(set(negative_answers))
            return negative_answers

        else:
            print("Invalid Pattern")
            exit()

    # The function used for finding a query that have at least one answer
    def sample_valid_question_with_answers(self, pattern):
        while True:
            _query = self.sample_with_pattern(pattern)
            _answers = self.query_search_answer(_query)
            if len(_answers) > 0:
                return _query, _answers


if __name__ == '__main__':
    n_queries_train_dict = n_queries_train_dict_small
    n_queries_valid_test_dict = n_queries_valid_test_dict_small

    first_round_query_types = {
        "2p": "(p,(p,(e)))",
        "3p": "(p,(p,(p,(e))))",
        "2i": "(i,(p,(e)),(p,(e)))",
        "3i": "(i,(p,(e)),(p,(e)),(p,(e)))",
        "ip": "(p,(i,(p,(e)),(p,(e))))",
        "pi": "(i,(p,(p,(e))),(p,(e)))",
        "2u": "(u,(p,(e)),(p,(e)))",
        "up": "(p,(u,(p,(e)),(p,(e))))",
    }

    one_p_query_types = {
        "1p": "(p,(e))"
    }

   
    for data_dir in n_queries_train_dict.keys():

        print("Load Train Graph " + data_dir)
        train_path = "./" + data_dir + "_train_with_units.pkl"
        train_graph = nx.read_gpickle(train_path)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in train_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("#nodes: ", train_graph.number_of_nodes())
        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
                reverse_attribute_edges_counter + numerical_edges_counter)

        print("Load Valid Graph " + data_dir)
        valid_path = "./" + data_dir +  "_valid_with_units.pkl"
        valid_graph = nx.read_gpickle(valid_path)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in valid_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("number of nodes: ", len(valid_graph.nodes))

        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
                reverse_attribute_edges_counter + numerical_edges_counter)

        print("Load Test Graph " + data_dir)
        test_path = "./" + data_dir + "_test_with_units.pkl"
        test_graph = nx.read_gpickle(test_path)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("number of nodes: ", len(test_graph.nodes))
        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
                reverse_attribute_edges_counter + numerical_edges_counter)

        # Print example edges:
        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
                print("example numerical edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, str):
                reverse_attribute_edges_counter += len(a)
                print("example reverse attribute edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(v, tuple) and isinstance(u, str):
                attribute_edges_counter += len(a)
                print("example attribute edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(v, str) and isinstance(u, str):
                attribute_edges_counter += len(a)
                print("example relation edge: ", u, v, a)
                break

       

        
        train_graph_sampler = GraphSamplerE34(train_graph)
        valid_graph_sampler = GraphSamplerE34(valid_graph)
        test_graph_sampler = GraphSamplerE34(test_graph)
        

        print("sample training queries")

        train_queries = {}
        valid_queries = {}
        test_queries = {}


        def sample_train_graph_with_pattern(pattern):
            while True:

                sampled_train_query = train_graph_sampler.sample_with_pattern(pattern)

                train_query_train_answers = train_graph_sampler.query_search_answer(sampled_train_query)
                if len(train_query_train_answers) > 0:
                    break
            return sampled_train_query, train_query_train_answers


        def sample_valid_graph_with_pattern(pattern):
            while True:

                sampled_valid_query = valid_graph_sampler.sample_with_pattern(pattern)

                valid_query_train_answers = train_graph_sampler.query_search_answer(sampled_valid_query)
                valid_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_valid_query)

                if len(valid_query_train_answers) > 0 and len(valid_query_valid_answers) > 0 \
                        and len(valid_query_train_answers) != len(valid_query_valid_answers):
                    break

            return sampled_valid_query, valid_query_train_answers, valid_query_valid_answers


        def sample_test_graph_with_pattern(pattern):
            while True:

                sampled_test_query = test_graph_sampler.sample_with_pattern(pattern)

                test_query_train_answers = train_graph_sampler.query_search_answer(sampled_test_query)
                test_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_test_query)
                test_query_test_answers = test_graph_sampler.query_search_answer(sampled_test_query)

                if len(test_query_train_answers) > 0 and len(test_query_valid_answers) > 0 \
                        and len(test_query_test_answers) > 0 and len(test_query_test_answers) != len(
                    test_query_valid_answers):
                    break
            return sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers



        this_type_train_queries = {}
        one_hop_query_list = train_graph_sampler.generate_one_p_queries()
        for one_hop_query in one_hop_query_list:
            train_one_hop_query_train_answers = train_graph_sampler.query_search_answer(one_hop_query)
            if len(train_one_hop_query_train_answers) > 0:
                this_type_train_queries[one_hop_query] = {"train_answers": train_one_hop_query_train_answers}

        train_queries["(p,(e))"] = this_type_train_queries

        with open(data_dir +  "_train_queries_1p.json", "w") as file_handle:
            json.dump(train_queries, file_handle)

        # for query_type, sample_pattern in first_round_query_types.items():

        #     print("train query_type: ", query_type)

        #     this_type_train_queries = {}

        #     n_query = 1

        #     for _ in tqdm(range(n_query)):
        #         sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern(sample_pattern)
        #         this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}

        #     train_queries[sample_pattern] = this_type_train_queries

        # with open(
        #         "../sampled_data/" + data_dir + "_train_queries.json",
        #         "w") as file_handle:
        #     json.dump(train_queries, file_handle)

        # validation_queries = {}
        # for query_type, sample_pattern in first_round_query_types.items():
        #     print("validation query_type: ", query_type)

        #     this_type_validation_queries = {}

        #     n_query = 1

           

        #     for _ in tqdm(range(n_query)):
        #         sampled_valid_query, valid_query_train_answers, valid_query_valid_answers = \
        #             sample_valid_graph_with_pattern(sample_pattern)

        #         this_type_validation_queries[sampled_valid_query] = {
        #             "train_answers": valid_query_train_answers,
        #             "valid_answers": valid_query_valid_answers
        #         }

        #     validation_queries[sample_pattern] = this_type_validation_queries


        # with open(
        #         "../sampled_data/" + data_dir  + "_valid_queries.json",
        #         "w") as file_handle:
        #     json.dump(validation_queries, file_handle)

        # test_queries = {}
        # for query_type, sample_pattern in first_round_query_types.items():
        #     print("test query_type: ", query_type)
        #     this_type_test_queries = {}

        #     n_query = 1

        
        #     for _ in tqdm(range(n_query)):
        #         sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers = \
        #             sample_test_graph_with_pattern(sample_pattern)

        #         this_type_test_queries[sampled_test_query] = {
        #             "train_answers": test_query_train_answers,
        #             "valid_answers": test_query_valid_answers,
        #             "test_answers": test_query_test_answers
        #         }

        #     test_queries[sample_pattern] = this_type_test_queries

        #     with open(
        #         "../sampled_data/" + data_dir +  "_test_queries.json",
        #         "w") as file_handle:
        #         json.dump(test_queries, file_handle)


            

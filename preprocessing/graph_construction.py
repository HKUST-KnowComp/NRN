"""
In experiment 2, the training graph is the original graph, a
"""
import networkx as nx

from random import sample, choice, randint, shuffle
import csv

from datetime import datetime as dt
from collections import Counter
from tqdm import tqdm
from copy import deepcopy



NUMERICAL_LIMIT = 50000

n_queries_train_dict_same = {
    "FB15K": 199408,
    "DB15K": 99363,
    "YAGO15K": 60750
}

n_queries_valid_test_dict_same = {
    "FB15K": 10000,
    "DB15K": 6000,
    "YAGO15K": 4000
}

n_edge_type = {
    "FB15K": 15,
    "DB15K": 30,
    "YAGO15K": 7
}


def toYearFraction(date):
    def sinceEpoch(_date):  # returns seconds since epoch
        return _date

    #         return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def GraphConstructor(edge_file_name, attribute_file_name, attribute_count_threshold):
    all_entities_dict = {}
    all_relation_dict = {}
    all_attribute_dict = {}

    attribute_name_list = []

    entity_tail_list = []
    attribute_tail_list = []

    # count the attributes first to determine which relations are to be remained
    with open(attribute_file_name, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')

        for row in spamreader:
            head, attribute_rel, attribute_value_raw = row[:3]
            attribute_name_list.append(attribute_rel)

    attribute_name_counter = Counter(attribute_name_list)
    top_attributes_with_freq = attribute_name_counter.most_common(attribute_count_threshold)
    top_attributes_names = [l[0] for l in top_attributes_with_freq]

    # convert the string to ids first
    with open(edge_file_name, "r") as file_in:

        spamreader = csv.reader(file_in, delimiter=' ')
        for row in spamreader:
            if row[0] not in all_entities_dict:
                all_entities_dict[row[0]] = str(len(all_entities_dict))

            if row[2] not in all_entities_dict:
                all_entities_dict[row[2]] = str(len(all_entities_dict))

            if row[1] not in all_relation_dict:
                all_relation_dict[row[1]] = len(all_relation_dict)

    with open(attribute_file_name, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')

        for row in spamreader:
            head, attribute_rel, attribute_value_raw = row[:3]

            if attribute_rel not in top_attributes_names:
                continue

            if head not in all_entities_dict:
                all_entities_dict[head] = str(len(all_entities_dict))

            if attribute_rel not in all_attribute_dict:
                all_attribute_dict[attribute_rel] = len(all_attribute_dict)

            attribute_name_list.append(attribute_rel)

    train_graph = nx.DiGraph()
    valid_graph = nx.DiGraph()
    test_graph = nx.DiGraph()

    # add edges from the entity relation triples

    relation_triple_list = []

    with open(edge_file_name, "r") as file_in:

        spamreader = csv.reader(file_in, delimiter=' ')
        for row in spamreader:
            assert len(row) == 4

            head_id = all_entities_dict[row[0]]
            tail_id = all_entities_dict[row[2]]

            relation_id = all_relation_dict[row[1]]

            relation_triple_list.append([head_id, tail_id, {relation_id * 2: 1}])

            entity_tail_list.append(tail_id)

    attribute_relation_list = []

    # Attribute Triples
    with open(attribute_file_name, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')

        for row in spamreader:
            head, attribute, attribute_value_raw = row[:3]

            if attribute not in top_attributes_names:
                continue

            head_id = all_entities_dict[head]
            attr_id = all_attribute_dict[attribute]

            if attribute not in top_attributes_names:
                continue

            dash_counter = Counter(attribute_value_raw)
            if dash_counter["-"] == 2:

                year_str, mont_str, day_str = attribute_value_raw.split("^^")[0].split("-")

                if "#" in year_str:
                    continue

                if mont_str == "0" or mont_str == "##":
                    mont_str = "1"
                if day_str == "0" or day_str == "##":
                    day_str = "1"

                while len(year_str) < 4:
                    year_str = "0" + year_str

                reconstruct_date_time = "-".join([year_str, mont_str, day_str])

                datam = dt.strptime(reconstruct_date_time, "%Y-%m-%d")
                numerical_value = toYearFraction(datam)
            elif dash_counter["-"] == 3:

                # Add 5000 years first
                _, year_str, mont_str, day_str = attribute_value_raw.split("^^")[0].split("-")

                if "#" in year_str:
                    continue

                if mont_str == "0" or mont_str == "##":
                    mont_str = "1"
                if day_str == "0" or day_str == "##":
                    day_str = "1"

                year_int = int(year_str)
                shifted_year = - year_int + 5000
                shifted_year = str(shifted_year)

                shifted_date_time = "-".join([shifted_year, mont_str, day_str])

                datam = dt.strptime(shifted_date_time, "%Y-%m-%d")

                numerical_value = toYearFraction(datam) - 5000

            else:
                numerical_value = float(attribute_value_raw.split("^^")[0])

            attribute_tail_list.append(numerical_value)
            attribute_relation_list.append([head_id, numerical_value, {attr_id * 2: 1}])

    entitiy_tail_counter = Counter(entity_tail_list)
    attribute_tail_counter = Counter(attribute_tail_list)

    # train, dev, test graph saparation

    shuffle(relation_triple_list)
    shuffle(attribute_relation_list)

    num_relation_edges = len(relation_triple_list)
    num_attribute_edges = len(attribute_relation_list)

    training_relation_edges = relation_triple_list[: int(0.8 * num_relation_edges)]
    training_attribute_edges = attribute_relation_list[: int(0.8 * num_attribute_edges)]

    validation_relation_edges = relation_triple_list[: int(0.9 * num_relation_edges)]
    validation_attribute_edges = attribute_relation_list[: int(0.9 * num_attribute_edges)]

    # add to graph, also add reverse edges

    def reverse_edge(edge_list):
        reversed_edges = []
        for h, t, a_dict in edge_list:
            reversed_a_dict = {}
            for k, v in a_dict.items():
                reversed_key = k + 1
                reversed_a_dict[reversed_key] = v
            reversed_edges.append([t, h, reversed_a_dict])
        return reversed_edges

    train_graph.add_edges_from(training_relation_edges)
    reversed_training_relation_edges = reverse_edge(training_relation_edges)
    train_graph.add_edges_from(reversed_training_relation_edges)

    train_graph.add_edges_from(training_attribute_edges)
    reversed_training_attribute_edges = reverse_edge(training_attribute_edges)
    train_graph.add_edges_from(reversed_training_attribute_edges)

    valid_graph.add_edges_from(validation_relation_edges)
    reversed_validation_relation_edges = reverse_edge(validation_relation_edges)
    valid_graph.add_edges_from(reversed_validation_relation_edges)

    valid_graph.add_edges_from(validation_attribute_edges)
    reversed_validation_attribute_edges = reverse_edge(validation_attribute_edges)
    valid_graph.add_edges_from(reversed_validation_attribute_edges)

    test_graph.add_edges_from(relation_triple_list)
    reversed_relation_triple_list = reverse_edge(relation_triple_list)
    test_graph.add_edges_from(reversed_relation_triple_list)

    test_graph.add_edges_from(attribute_relation_list)
    reversed_attribute_relation_list = reverse_edge(attribute_relation_list)
    test_graph.add_edges_from(reversed_attribute_relation_list)

    return train_graph, valid_graph, test_graph, entitiy_tail_counter, attribute_tail_counter


def approximately_equal(x_1, x_2):
    difference = np.abs(x_1 - x_2)
    absolute_x_1 = np.abs(x_1)
    absolute_x_2 = np.abs(x_2)

    if ((difference / absolute_x_1) < 0.01) and ((difference / absolute_x_2) < 0.01):
        return True
    return False


def greater_than(x_1, x_2):
    return x_1 > x_2


def smaller_than(x_1, x_2):
    return x_1 < x_2


def approximately_two_times_equal_to(x_1, x_2):
    return approximately_equal(2 * x_1, x_2)


def approximately_three_times_equal_to(x_1, x_2):
    return approximately_equal(3 * x_1, x_2)


def two_times_larger_than(x_1, x_2):
    return greater_than(x_1, 2 * x_2)


def three_times_larger_than(x_1, x_2):
    return greater_than(x_1, 3 * x_2)


numerical_constraints_dict = {
    "ae": approximately_equal,
    "gt": greater_than,
    "st": smaller_than,
    "a2e": approximately_two_times_equal_to,
    "a3e": approximately_three_times_equal_to,
    "2gt": two_times_larger_than,
    "3gt": three_times_larger_than,
}


def graph_constraint_counter(func, graph):
    counter = 0

    values_sets = {}

    for u, v, a in tqdm(graph.edges(data=True)):
        if isinstance(u, float) and isinstance(v, float):
            continue

        if isinstance(v, float):
            for attr in a.keys():
                if attr in values_sets:
                    values_sets[attr].append(v)
                else:
                    values_sets[attr] = [v]
    for attr, value_set in tqdm(values_sets.items()):
        for u in value_set:
            if isinstance(u, float):
                for v in value_set:
                    if isinstance(v, float):
                        if func(u, v):
                            counter += 1
    return counter


def graph_add_constraints(func, graph, edge_id):
    added_edges = []

    values_sets = {}

    for u, v, a in tqdm(graph.edges(data=True)):
        if isinstance(u, float) and isinstance(v, float):
            continue

        if isinstance(v, float):
            for attr in a.keys():
                if attr in values_sets:
                    values_sets[attr].append(v)
                else:
                    values_sets[attr] = [v]

    for attr, value_set in tqdm(values_sets.items()):
        counter = 0

        for u in value_set:
            if counter >= NUMERICAL_LIMIT // len(values_sets):
                break
            if isinstance(u, float):
                for v in value_set:
                    if counter >= NUMERICAL_LIMIT // len(values_sets):
                        break

                    if isinstance(v, float):
                        if func(u, v):
                            added_edges.append([u, v, {edge_id: 1}])
                            counter += 1

    graph.add_edges_from(added_edges)


if __name__ == '__main__':
    n_queries_train_dict = n_queries_train_dict_same
    n_queries_valid_test_dict = n_queries_valid_test_dict_same

    first_round_query_types = {
        "1p": "(p,(e))",
        "2p": "(p,(p,(e)))",
        "2i": "(i,(p,(e)),(p,(e)))",
        "3i": "(i,(p,(e)),(p,(e)),(p,(e)))",
        "ip": "(p,(i,(p,(e)),(p,(e))))",
        "pi": "(i,(p,(p,(e))),(p,(e)))",
        "2u": "(u,(p,(e)),(p,(e)))",
        "up": "(p,(u,(p,(e)),(p,(e))))",
        "2in": "(i,(n,(p,(e))),(p,(e)))",
        "3in": "(i,(n,(p,(e))),(p,(e)),(p,(e)))",
        "inp": "(p,(i,(n,(p,(e))),(p,(e))))",
        "pin": "(i,(p,(p,(e))),(n,(p,(e))))",
        "pni": "(i,(n,(p,(p,(e)))),(p,(e)))",
    }

    for data_dir in n_queries_train_dict.keys():

        import numpy as np

        train_graph, valid_graph, test_graph, entitiy_tail_counter, attribute_tail_counter = \
            GraphConstructor("../data/" + data_dir + "/" + data_dir + "_EntityTriples.txt",
                             "../data/" + data_dir + "/" + data_dir + "_NumericalTriples.txt",
                             n_edge_type[data_dir])

        experiment_2_train_graph = deepcopy(train_graph)
        experiment_2_valid_graph = deepcopy(valid_graph)
        experiment_2_test_graph = deepcopy(test_graph)

        experiment_3_4_train_graph = deepcopy(train_graph)
        experiment_3_4_valid_graph = deepcopy(valid_graph)
        experiment_3_4_test_graph = deepcopy(test_graph)

        # The test graph of experiment 2 only add approximately equal edges for sampling test edges
        graph_add_constraints(approximately_equal, experiment_2_test_graph, 0)
        graph_add_constraints(approximately_equal, experiment_2_valid_graph, 0)

        # The train, valid, and test graphs are all added with all numerical edges
        for graph in [experiment_3_4_train_graph, experiment_3_4_valid_graph, experiment_3_4_test_graph]:
            for id, func in enumerate(numerical_constraints_dict.values()):
                graph_add_constraints(func, graph, id)

        graph_names = ["e2_train.pkl", "e2_valid.pkl", "e2_test.pkl",
                       "e34_train.pkl", "e34_valid.pkl", "e34_test.pkl"]
        for graph_id, graph in enumerate([experiment_2_train_graph, experiment_2_valid_graph, experiment_2_test_graph,
                                          experiment_3_4_train_graph, experiment_3_4_valid_graph,
                                          experiment_3_4_test_graph]):

            print("#nodes: ", graph.number_of_nodes())

            entity_counter = 0
            value_counter = 0
            for u in graph.nodes():
                res = isinstance(u, str)
                if res:
                    entity_counter += 1
                else:
                    value_counter += 1

            print("#entity nodes", entity_counter)
            print("#attribute nodes", value_counter)

            edge_counter = 0
            for u, v, a in graph.edges(data=True):
                edge_counter += len(a)

            print("#head&tails: ", graph.number_of_edges())
            print("#edges: ", edge_counter)

            one_p_list = []
            for u, v, a in graph.edges(data=True):
                for attr in a.keys():
                    one_p_list.append((u, attr))

            one_p_set = set(one_p_list)
            print("#1p: ", len(one_p_set))

            print("#average entity in-degree", np.array(list(entitiy_tail_counter.values())).mean())
            print("#average attribute value in-degree", np.array(list(attribute_tail_counter.values())).mean())
            print(type(list(entitiy_tail_counter.keys())[0]))
            print(type(list(attribute_tail_counter.keys())[0]))

            nx.write_gpickle(graph, data_dir + "_" + graph_names[graph_id])
            # for constraints_name, constraints_func in numerical_constraints_dict.items():
            #     print(constraints_name)
            #     print(graph_constraint_counter(constraints_func, graph))

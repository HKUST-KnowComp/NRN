import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator, separate_query_dict
import networkx as nx


class GeneralModel(nn.Module):

    def __init__(self, num_entities, num_relations, embedding_size):
        super(GeneralModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size

    def loss_fnt(self, sub_query_encoding, labels):
        raise NotImplementedError

    def scoring(self, query_encoding):
        """
        :param query_encoding:
        :return: [batch_size, num_entities]
        """
        raise NotImplementedError

    def evaluate_entailment(self, query_encoding, entailed_answers):
        # print("entailment evaluation")
        """
        We do not have to conduct the evaluation on GPU as it is not necessary.


        :param query_encoding:
        :param entailed_answers:
        :return:
        """

        

        if isinstance(entailed_answers[0][0], tuple):
            # Evaluate a numerical query here
            learned_value = self.decoder(query_encoding)[:, self.num_entities:]
            
            log_list = []

            for i in range(len(entailed_answers)):
                # [num_attribute_values]
                # all_scoring = self.individual_scoring_value(query_encoding[i], id2value_list)

                if len(self.number_encoder_list) == 1:

                   

                    label_ids = self.number_encoder_list[0].values2ids([l[0] for l in entailed_answers[i]])
                    
                    all_value_embeddings = self.number_encoder_list[0].get_embeddings()
                
                else:

                    label_ids = []

                    for label in entailed_answers[i]:

                        label_offset = self.number_encoder_offsets[label[1]]
                        # print("label_offset", label_offset)
                        label_id = label_offset +  self.number_encoder_list[label[1]].value2id(label[0])
                        # print("label_id", label_id)

                        label_ids.append(label_id)

                    typed_value_embeddings = []
                    for encoder_id, encoder in  enumerate(self.number_encoder_list):
                        value_embedding = encoder.get_embeddings()
                        encoder_id = torch.tensor([encoder_id]).to(value_embedding.device)
                        value_type_embedding = self.number_type_embedding(encoder_id)
                        
                        typed_value_embeddings.append(value_embedding + value_type_embedding)

                    all_value_embeddings = torch.cat(typed_value_embeddings, dim=0)


                
                query_encoding_scores = torch.matmul(query_encoding[i], all_value_embeddings.t())

                if self.mixed_value_reprerentation:
                
                    learned_number_encoding_scores = learned_value[i]

                    
                    query_encoding_scores = query_encoding_scores + learned_number_encoding_scores



                #  [num_attribute_values]
                all_scoring = query_encoding_scores
                original_scores = query_encoding_scores.clone()


                answer_ids = label_ids
                answer_ids = torch.LongTensor(answer_ids)
                all_scoring[answer_ids] = - 10000000

                # [num_answers]
                answer_scores = original_scores[answer_ids]
                
                # [num_answers, 1]
                answer_scores = answer_scores.unsqueeze(1)

                # [1, num_labels]
                all_scoring = all_scoring.unsqueeze(0)


                # [num_entailed_answers, num_entities]
                answer_is_smaller_matrix = ((answer_scores - all_scoring) < 0)

                # [num_entailed_answers]
                answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                # [num_entailed_answers]
                rankings = answer_rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                num_answers = len(entailed_answers[i])

                logs = {
                    "ent_mrr": mrr,
                    "ent_hit_at_1": hit_at_1,
                    "ent_hit_at_3": hit_at_3,
                    "ent_hit_at_10": hit_at_10,
                    "ent_num_answers": num_answers
                }

                log_list.append(logs)
            return log_list
            
        else:
            # Evaluate a entity query here
            # [batch_size, num_entities]
            all_scoring = self.scoring(query_encoding)

            # [batch_size, num_entities]
            original_scores = all_scoring.clone()

            log_list = []

            for i in range(len(entailed_answers)):
                entailed_answer_set = torch.tensor(entailed_answers[i])

                # [num_entities]
                not_answer_scores = all_scoring[i]
                not_answer_scores[entailed_answer_set] = - 10000000

                # [1, num_entities]
                not_answer_scores = not_answer_scores.unsqueeze(0)

                # [num_entailed_answers, 1]
                entailed_answers_scores = original_scores[i][entailed_answer_set].unsqueeze(1)

                # [num_entailed_answers, num_entities]
                answer_is_smaller_matrix = ((entailed_answers_scores - not_answer_scores) < 0)

                # [num_entailed_answers]
                answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                # [num_entailed_answers]
                rankings = answer_rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                num_answers = len(entailed_answers[i])

                logs = {
                    "ent_mrr": mrr,
                    "ent_hit_at_1": hit_at_1,
                    "ent_hit_at_3": hit_at_3,
                    "ent_hit_at_10": hit_at_10,
                    "ent_num_answers": num_answers
                }

                log_list.append(logs)
            return log_list

    def evaluate_generalization(self, query_encoding, entailed_answers, generalized_answers):

        """

        This function is largely different from the evaluation of previous work, and we conduct a more rigorous
        evaluation. In previous methods, when it contains existential positive queries without negations, we can all
        the answers in the graph that are entailed ``easy'' answers. But when it includes the negation operations,
        the answers entailed by the training graph may not be valid answers anymore !!! This is very critical in terms
        of evaluating the queries with negation/exclusion , but is ignored in all relevant work. Without evaluating
        the answers that are excluded by the reasoning, we cannot evaluate the actual performance of complement.

        :param query_encoding:
        :param entailed_answers:
        :param generalized_answers:
        :return:
        """

        if isinstance(entailed_answers[0][0], tuple):
            # Evaluate a numerical query here
            
            learned_value = self.decoder(query_encoding)[:, self.num_entities:]

            log_list = []

            for i in range(len(entailed_answers)):
                # [num_labels]

                # entailed_ids = self.number_encoder.values2ids(entailed_answers[i])
                # all_value_embeddings = self.number_encoder.get_embeddings()


                if len(self.number_encoder_list) == 1:
                   
                    all_value_embeddings = self.number_encoder_list[0].get_embeddings()
                
                else:
                    
                    
                    typed_value_embeddings = []
                    for encoder_id, encoder in  enumerate(self.number_encoder_list):
                        value_embedding = encoder.get_embeddings()
                        encoder_id = torch.tensor([encoder_id]).to(value_embedding.device)
                        value_type_embedding = self.number_type_embedding(encoder_id)
                        
                        typed_value_embeddings.append(value_embedding + value_type_embedding)

                    all_value_embeddings = torch.cat(typed_value_embeddings, dim=0)
                            

                    

                query_encoding_scores = torch.matmul(query_encoding[i], all_value_embeddings.t())

                if self.mixed_value_reprerentation:
                    learned_number_encoding_scores = learned_value[i]
                    query_encoding_scores = query_encoding_scores + learned_number_encoding_scores




                #  [num_attribute_values]
                all_scoring = query_encoding_scores
                original_scores = query_encoding_scores.clone()

                # print("all_scoring",all_scoring.shape)
                # print("",all_scoring)

                all_answers = list(set(entailed_answers[i]) | set(generalized_answers[i]))
                need_to_inferred_answers = list(set(generalized_answers[i]) - set(entailed_answers[i]))

                if len(self.number_encoder_list) == 1:
                    all_answers_ids = self.number_encoder_list[0].values2ids([l[0] for l in all_answers])
                else:

                    all_answers_ids = []

                    for label in all_answers:

                        label_offset = self.number_encoder_offsets[label[1]]
                        # print("label_offset", label_offset)
                        label_id = label_offset +  self.number_encoder_list[label[1]].value2id(label[0])
                        # print("label_id", label_id)

                        all_answers_ids.append(label_id)

                        

                all_scoring[all_answers_ids] = - 10000000

                logs = {}

                if len(need_to_inferred_answers) > 0:
                    num_answers = len(need_to_inferred_answers)

                    if len(self.number_encoder_list) == 1:
                        label_ids = self.number_encoder_list[0].values2ids([l[0] for l in need_to_inferred_answers])
                        
                       
                    
                    else:
                        label_ids = []

                        for label in need_to_inferred_answers:

                            label_offset = self.number_encoder_offsets[label[1]]
                            # print("label_offset", label_offset)
                            label_id = label_offset +  self.number_encoder_list[label[1]].value2id(label[0])
                            # print("label_id", label_id)

                            label_ids.append(label_id)
                    

                    label_ids = torch.LongTensor(label_ids)

                    # [num_need_to_inferred_answers, 1]
                    need_to_inferred_answers_scores = original_scores[label_ids].unsqueeze(1)

                    # [num_need_to_inferred_answers, num_entities]
                    answer_is_smaller_matrix = ((need_to_inferred_answers_scores - all_scoring) < 0)

                    # [num_need_to_inferred_answers]
                    answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                    # [num_need_to_inferred_answers]
                    rankings = answer_rankings.float()

                    mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                    hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                    hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                    hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                    logs["inf_mrr"] = mrr
                    logs["inf_hit_at_1"] = hit_at_1
                    logs["inf_hit_at_3"] = hit_at_3
                    logs["inf_hit_at_10"] = hit_at_10
                    logs["inf_num_answers"] = num_answers
                else:
                    logs["inf_mrr"] = 0
                    logs["inf_hit_at_1"] = 0
                    logs["inf_hit_at_3"] = 0
                    logs["inf_hit_at_10"] = 0
                    logs["inf_num_answers"] = 0


                log_list.append(logs)
            return log_list

        else:
            # [batch_size, num_entities]
            all_scoring = self.scoring(query_encoding)

            # [batch_size, num_entities]
            original_scores = all_scoring.clone()

            log_list = []

            for i in range(len(entailed_answers)):

                all_answers = list(set(entailed_answers[i]) | set(generalized_answers[i]))
                need_to_exclude_answers = list(set(entailed_answers[i]) - set(generalized_answers[i]))
                need_to_inferred_answers = list(set(generalized_answers[i]) - set(entailed_answers[i]))

                all_answers_set = torch.tensor(all_answers)

                # [num_entities]
                not_answer_scores = all_scoring[i]
                not_answer_scores[all_answers_set] = - 10000000

                # [1, num_entities]
                not_answer_scores = not_answer_scores.unsqueeze(0)

                logs = {}

                if len(need_to_inferred_answers) > 0:
                    num_answers = len(need_to_inferred_answers)

                    need_to_inferred_answers = torch.tensor(need_to_inferred_answers)

                    # [num_need_to_inferred_answers, 1]
                    need_to_inferred_answers_scores = original_scores[i][need_to_inferred_answers].unsqueeze(1)

                    # [num_need_to_inferred_answers, num_entities]
                    answer_is_smaller_matrix = ((need_to_inferred_answers_scores - not_answer_scores) < 0)

                    # [num_need_to_inferred_answers]
                    answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

                    # [num_need_to_inferred_answers]
                    rankings = answer_rankings.float()

                    mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                    hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                    hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                    hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                    logs["inf_mrr"] = mrr
                    logs["inf_hit_at_1"] = hit_at_1
                    logs["inf_hit_at_3"] = hit_at_3
                    logs["inf_hit_at_10"] = hit_at_10
                    logs["inf_num_answers"] = num_answers
                else:
                    logs["inf_mrr"] = 0
                    logs["inf_hit_at_1"] = 0
                    logs["inf_hit_at_3"] = 0
                    logs["inf_hit_at_10"] = 0
                    logs["inf_num_answers"] = 0


                log_list.append(logs)
            return log_list


class IterativeModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size):
        super(IterativeModel, self).__init__(num_entities, num_relations, embedding_size)

    def projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def relation_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def attribute_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def reversed_attribute_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def numerical_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def higher_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def intersection(self, sub_query_encoding_list):
        raise NotImplementedError

    def union(self, sub_query_encoding_list):
        raise NotImplementedError

    def number_encoding(self, values):
        raise NotImplementedError

    def negation(self, sub_query_encoding):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):

        assert batched_structured_query[0] in ["p", "e", "i", "u", "n", "ap", "rp", "rap", "np", "nv"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "ap":

            sub_query_result = self.forward(batched_structured_query[2])
            this_query_result = self.attribute_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "rap":

            sub_query_result = self.forward(batched_structured_query[2])
            this_query_result = self.reversed_attribute_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "rp":

            sub_query_result = self.forward(batched_structured_query[2])
            this_query_result = self.relation_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "np":
            sub_query_result = self.forward(batched_structured_query[2])
            this_query_result = self.numerical_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.intersection(sub_query_result_list)

        elif batched_structured_query[0] == "u":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.union(sub_query_result_list)

        elif batched_structured_query[0] == "n":
            sub_query_result = self.forward(batched_structured_query[1])
            this_query_result = self.negation(sub_query_result)

        elif batched_structured_query[0] == "e":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_result = self.entity_embedding(entity_ids)

        elif batched_structured_query[0] == "nv":

            entity_values = torch.tensor(batched_structured_query[1])
            entity_values = entity_values.to(self.entity_embedding.weight.device)
            this_query_result = self.number_encoding(entity_values)

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            return self.loss_fnt(this_query_result, label)


class SequentialModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size):
        super().__init__(num_entities, num_relations, embedding_size)

    def encode(self, batched_structured_query):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):

        batched_structured_query = torch.tensor(batched_structured_query)
        if torch.cuda.is_available():
            batched_structured_query = batched_structured_query.cuda()

        representations = self.encode(batched_structured_query)

        if label is not None:
            return self.loss_fnt(representations, label)

        else:
            return representations


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)



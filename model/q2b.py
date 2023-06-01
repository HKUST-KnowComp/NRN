import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import IterativeModel, LabelSmoothingLoss

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator, separate_query_dict
import pickle
import networkx as nx
import numpy as np
import math

from numeral_encoder import PositionalEncoder, DICE, GMM_Prototype, DigitRNN


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):
        # [batch_size, num_particles, embedding_size]
        K = self.query(particles)
        V = self.query(particles)
        Q = self.query(particles)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output


class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """

    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class ParticleCrusher(nn.Module):

    def __init__(self, embedding_size, num_particles):
        super(ParticleCrusher, self).__init__()

        # self.noise_layer = nn.Linear(embedding_size, embedding_size)
        self.num_particles = num_particles

        self.off_sets = nn.Parameter(torch.zeros([1, num_particles, embedding_size]), requires_grad=True)
        # self.layer_norm = LayerNorm(embedding_size)

    def forward(self, batch_of_embeddings):
        # shape of batch_of_embeddings: [batch_size, embedding_size]
        # the return is a tuple ([batch_size, embedding_size, num_particles], [batch_size, num_particles])
        # The first return is the batch of particles for each entity, the second is the weights of the particles
        # Use gaussian kernel to do this

        batch_size, embedding_size = batch_of_embeddings.shape

        # [batch_size, num_particles, embedding_size]
        expanded_batch_of_embeddings = batch_of_embeddings.reshape(batch_size, -1, embedding_size) + self.off_sets

        return expanded_batch_of_embeddings

class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class Q2B(IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, gamma=12, alpha=0.02, number_encoder_list=None, label_smoothing=0.1, 
                 numerical_label_smoothing=0.3,
                 dropout_rate=0.3, num_attributes = None, 
                 num_numrical_proj = None, value_vocab=None, mixed_value_reprerentation=False):
        super(Q2B, self).__init__(num_entities, num_relations, embedding_size)

        # initialize embeddings
        # we treat entities as boxes with offset=0

        if not mixed_value_reprerentation:
            self.entity_embedding = nn.Embedding(num_entities, embedding_size)
            embedding_weights = self.entity_embedding.weight
            self.decoder = nn.Linear(embedding_size,
                                    num_entities,
                                    bias=False)
            self.decoder.weight = embedding_weights
        else:
            self.entity_embedding = nn.Embedding(num_entities + len(value_vocab), embedding_size)

            embedding_weights = self.entity_embedding.weight
            self.decoder = nn.Linear(embedding_size,
                                    num_entities + len(value_vocab),
                                    bias=False)
            self.decoder.weight = embedding_weights

        self.relation_center_embedding = nn.Embedding(num_relations, embedding_size)
        self.relation_offset_embedding = nn.Embedding(num_relations, embedding_size)


        if number_encoder_list is not None and len(number_encoder_list) > 1:
            self.number_type_embedding = nn.Embedding(len(number_encoder_list), embedding_size)
        

        self.value_vocab = value_vocab

        self.mixed_value_reprerentation = mixed_value_reprerentation

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()




        self.embedding_size = embedding_size
        # box intersection nets

        self.center_intersection_net = CenterIntersection(self.embedding_size)
        self.offset_intersection_net = BoxOffsetIntersection(self.embedding_size)

        self.center_union_net = CenterIntersection(self.embedding_size)
        self.offset_union_net = BoxOffsetIntersection(self.embedding_size)

        self.gamma = nn.Parameter(  # Margin when calculating score
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.alpha = nn.Parameter(  # ratio to discount in-box distance
            torch.Tensor([alpha]),
            requires_grad=False
        )

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        self.numerical_label_smoothing_loss = LabelSmoothingLoss(smoothing=numerical_label_smoothing)

        self.number_encoder_list = number_encoder_list

        self.number_encoder_offsets = []

        number_counter = 0
        if self.number_encoder_list is not None:
            for encoder in self.number_encoder_list:
                self.number_encoder_offsets.append(number_counter)
                number_counter += len(encoder.all_values)
    


        # other projection weights
        # Projection weights
        self.projection_layer_norm_1 = LayerNorm(embedding_size)
        self.projection_layer_norm_2 = LayerNorm(embedding_size)

        self.projection_self_attn = SelfAttention(embedding_size)

        self.projection_Wz = nn.Linear(embedding_size, embedding_size)
        self.projection_Uz = nn.Linear(embedding_size, embedding_size)

        self.projection_Wr = nn.Linear(embedding_size, embedding_size)
        self.projection_Ur = nn.Linear(embedding_size, embedding_size)

        self.projection_Wh = nn.Linear(embedding_size, embedding_size)
        self.projection_Uh = nn.Linear(embedding_size, embedding_size)


        # Attribute projection weights
        self.attribute_proj_layer_norm = LayerNorm(embedding_size)
        

        # Reversed attribute projection weights
        self.reversed_attribute_proj_layer_norm = LayerNorm(embedding_size)
        

        # numerical projection weights
        self.numerical_proj_layer_norm = LayerNorm(embedding_size)
      

        if num_attributes is not None:
            self.attribute_embedding = nn.Embedding(num_attributes * 2, embedding_size)

        if num_numrical_proj is not None:
            self.numerical_op_embedding = nn.Embedding(num_numrical_proj, embedding_size)

        self.numerical_intersection_attn = SelfAttention(embedding_size)
        self.numerical_intersection_ffn = FFN(embedding_size, self.dropout)
        self.numerical_intersection_layer_norm = LayerNorm(embedding_size)


        self.numerical_union_attn = SelfAttention(embedding_size)
        self.numerical_union_ffn = FFN(embedding_size, self.dropout)
        self.numerical_union_layer_norm = LayerNorm(embedding_size) 

        # Crusher
        self.to_particles = ParticleCrusher(embedding_size, 2) 



    # def scoring(self, query_box_encoding):
    #     """

    #     :param query_box_encoding: ([batch_size, embedding_size],[batch_size, embedding_size])
    #     :return: [batch_size, num_entities]
    #     """

    #     # [num_entities, embedding_size]
    #     entity_embeddings = self.entity_embedding.weight
    #     query_center_embedding, query_offset_embedding = query_box_encoding

    #     """
    #     [14505* 300] -> [5*14505*300] -> [14505*5*300]  .repeat(len(query_box_encoding[0]),1,1).permute([1,0,2])
    #     [5* 300] -> [14505*5*300]     .repeat(len(entity_embeddings),1,1)
    #     """
      


    #     # [1, num_entities, embedding_size]
    #     enlarged_entity_embeddings = entity_embeddings.unsqueeze(0)

    #     # [batch_size, 1, embedding_size]
    #     enlarged_center_embeddings = query_center_embedding.unsqueeze(1)

    #     # [batch_size, 1, embedding_size]
    #     enlarged_offset_embeddings = query_offset_embedding.unsqueeze(1)

    #     q_max = enlarged_center_embeddings + enlarged_offset_embeddings
    #     q_min = enlarged_center_embeddings - enlarged_offset_embeddings

    #     # [batch_size, num_entities]
    #     dist_out = (F.relu(enlarged_entity_embeddings - q_max) + F.relu(q_min - enlarged_entity_embeddings)).sum(dim=-1)

    #     dist_in = (enlarged_center_embeddings - torch.minimum( q_max, torch.maximum(q_min, enlarged_entity_embeddings))).abs().sum(dim=-1)

       
    #     distances = dist_out + self.alpha * dist_in

    #     return - distances


    def scoring(self, query_box_encoding):
        """

        :param query_box_encoding: ([batch_size, embedding_size],[batch_size, embedding_size])
        :return: [batch_size, num_entities]
        """

        # [num_entities, embedding_size]
        entity_embeddings = self.entity_embedding.weight

        # [batch_size, embedding_size]
        query_center_embedding, query_offset_embedding = query_box_encoding

        """
        [14505* 300] -> [5*14505*300] -> [14505*5*300]  .repeat(len(query_box_encoding[0]),1,1).permute([1,0,2])
        [5* 300] -> [14505*5*300]     .repeat(len(entity_embeddings),1,1)
        """

        # [batch_size, num_entities]
        similarity_in = torch.matmul(query_center_embedding, entity_embeddings.transpose(0,1))

        # considering the inner product of is liner operation to both operands, 
        # also considering query_offset_embedding is a non-negative vector, 
        # we can use the following formula to compute the difference of inner product
        similarity_delta = torch.matmul(query_offset_embedding, torch.abs(entity_embeddings).transpose(0,1))

        similarity_out = similarity_in + similarity_delta

        similarity = similarity_out + self.alpha * similarity_in
        

        return similarity
    def projection(self, relation_ids, sub_query_box_embedding):
        """
        The relational projection of query2box

        :param relation_ids: [batch_size]
        :param sub_query_center_embedding: [batch_size, embedding_size]
        :param sub_query_offset_embedding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size], [batch_size, embedding_size] (center + offset)
        """
        # print(len(sub_query_box_embedding))
        sub_query_center_embedding, sub_query_offset_embedding = sub_query_box_embedding
        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(
            self.relation_center_embedding.weight.device)  # What's the usage of this sentence?

        relation_center_embeddings = self.relation_center_embedding(relation_ids)
        relation_offset_embeddings = self.relation_offset_embedding(relation_ids)

        new_center_embedding = relation_center_embeddings + sub_query_center_embedding
        new_offset_embedding = relation_offset_embeddings + sub_query_offset_embedding
        new_box_embedding = tuple([new_center_embedding, new_offset_embedding])

        return new_box_embedding

    def higher_projection(self, relation_ids, sub_query_box_embedding):
        return self.projection(relation_ids, sub_query_box_embedding)

    

    def relation_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)

    def attribute_projection(self, attribute_ids, sub_query_encoding):

        # print("attribute_projection input shape:", sub_query_encoding.shape)

        # [batch_size, embedding_size]

        Wz = self.projection_Wz
        Uz = self.projection_Uz

        Wr = self.projection_Wr
        Ur = self.projection_Ur

        Wh = self.projection_Wh
        Uh = self.projection_Uh


        attribute_ids = torch.tensor(attribute_ids)
        attribute_ids = attribute_ids.to(self.attribute_embedding.weight.device)

        attribute_embeddings = self.attribute_embedding(attribute_ids)


        relation_transition = torch.unsqueeze(attribute_embeddings, 1)

        projected_particles = torch.stack(sub_query_encoding, dim=1)



        z = self.sigmoid(Wz(self.dropout(relation_transition)) + Uz(self.dropout(projected_particles)))
        r = self.sigmoid(Wr(self.dropout(relation_transition)) + Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(Wh(self.dropout(relation_transition)) + Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * projected_particles + z * h_hat

        projected_particles = h
        projected_particles = self.projection_layer_norm_1(projected_particles)

        projected_particles = self.projection_self_attn(self.dropout(projected_particles)).sum(dim=1)
        projected_particles = self.attribute_proj_layer_norm(projected_particles)


        # positions = self.attribute_proj_layer_norm(attribute_embeddings + sub_query_encoding.sum(dim=1))

        # print("attribute_projection output shape:", positions.shape)

        return projected_particles

    def reversed_attribute_projection(self, attribute_ids, sub_query_encoding):

        # [batch_size, embedding_size]
        # print("reversed_attribute_projection input shape:", sub_query_encoding.shape)

        attribute_ids = torch.tensor(attribute_ids)
        attribute_ids = attribute_ids.to(self.attribute_embedding.weight.device)

        attribute_embeddings = self.attribute_embedding(attribute_ids)

        sub_query_encoding = self.to_particles(sub_query_encoding)


        Wz = self.projection_Wz
        Uz = self.projection_Uz

        Wr = self.projection_Wr
        Ur = self.projection_Ur

        Wh = self.projection_Wh
        Uh = self.projection_Uh


        relation_transition = torch.unsqueeze(attribute_embeddings, 1)

        projected_particles = sub_query_encoding



        z = self.sigmoid(Wz(self.dropout(relation_transition)) + Uz(self.dropout(projected_particles)))
        r = self.sigmoid(Wr(self.dropout(relation_transition)) + Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(Wh(self.dropout(relation_transition)) + Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * projected_particles + z * h_hat

        projected_particles = h
        projected_particles = self.projection_layer_norm_1(projected_particles)

        projected_particles = self.projection_self_attn(self.dropout(projected_particles))
        projected_particles = self.reversed_attribute_proj_layer_norm(projected_particles)
        
       
        # all_embeddings = attribute_embeddings + sub_query_encoding

        # all_embeddings = self.reversed_attribute_proj_layer_norm(all_embeddings)

        # projected_encodings = self.reversed_attribute_proj(all_embeddings).reshape(all_embeddings.shape[0], -1, self.embedding_size)
        # projected_encodings = self.reversed_attribute_proj_layer_norm(projected_encodings)

        # print("reversed_attribute_projection output shape:", projected_encodings.shape)
        projected_particles = [projected_particles[:, i, :] for i in range(projected_particles.shape[1]) ]
        projected_particles = tuple(projected_particles)

        return projected_particles

    def numerical_projection(self, numerical_proj_ids, sub_query_encoding):

        # print("reversed_attribute_projection input shape:", sub_query_encoding.shape)
        
        numerical_proj_ids = torch.tensor(numerical_proj_ids)
        numerical_proj_ids = numerical_proj_ids.to(self.numerical_op_embedding.weight.device)




        numerical_embeddings = self.numerical_op_embedding(numerical_proj_ids)
        


        # print("reversed_attribute_projection output shape:", numerical_embeddings.shape)


        sub_query_encoding = self.to_particles(sub_query_encoding)


        Wz = self.projection_Wz
        Uz = self.projection_Uz

        Wr = self.projection_Wr
        Ur = self.projection_Ur

        Wh = self.projection_Wh
        Uh = self.projection_Uh


        relation_transition = torch.unsqueeze(numerical_embeddings, 1)

        projected_particles = sub_query_encoding



        z = self.sigmoid(Wz(self.dropout(relation_transition)) + Uz(self.dropout(projected_particles)))
        r = self.sigmoid(Wr(self.dropout(relation_transition)) + Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(Wh(self.dropout(relation_transition)) + Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * projected_particles + z * h_hat

        projected_particles = h
        projected_particles = self.projection_layer_norm_1(projected_particles)

        projected_particles = self.projection_self_attn(self.dropout(projected_particles)).sum(dim=1)

        projected_particles = self.numerical_proj_layer_norm(projected_particles)

        
        return projected_particles




    def intersection(self, sub_query_box_embedding_list):
        """
        :param: sub_query_box_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
        :return:  [batch_size, embedding_size], [batch_size, embedding_size]
        """

        if isinstance(sub_query_box_embedding_list[0], list):
            # Intersection of boxes

            sub_query_center_embedding_list, sub_query_offset_embedding_list = sub_query_box_embedding_list

            
            

            """
            [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]

            after intersection

            [a12,b12,c12,d12,e12] 
            """
            all_subquery_center_encodings = torch.stack(sub_query_center_embedding_list, dim=0)
            all_subquery_offset_encodings = torch.stack(sub_query_offset_embedding_list, dim=0)

            new_query_center_embeddings = self.center_intersection_net(all_subquery_center_encodings)
            new_query_offset_embeddings = self.offset_intersection_net(all_subquery_offset_encodings)

            new_query_box_embeddings = tuple([new_query_center_embeddings, new_query_offset_embeddings])
            return new_query_box_embeddings
        
        else:
            # Intersection of values
            all_subquery_encodings = torch.stack(sub_query_box_embedding_list, dim=1)
            batch_size, num_sets, embedding_size = all_subquery_encodings.shape

            flatten_particles = all_subquery_encodings.view(batch_size, -1, embedding_size)


            flatten_particles = self.numerical_intersection_attn(self.dropout(flatten_particles))
            flatten_particles = self.numerical_intersection_layer_norm(flatten_particles)
            flatten_particles = self.numerical_intersection_ffn(flatten_particles) + flatten_particles
            flatten_particles = self.numerical_intersection_layer_norm(flatten_particles)

            encoding = flatten_particles.sum(dim=1)
            
            return encoding


    
    def union(self, sub_query_box_embedding_list):
        """
        :param: sub_query_box_embedding_list (tuple of two list of size [num_sub_queries, batch_size, embedding_size])
        :return:  [batch_size, embedding_size], [batch_size, embedding_size]
        """

        if isinstance(sub_query_box_embedding_list[0], list):
            sub_query_center_embedding_list, sub_query_offset_embedding_list = sub_query_box_embedding_list
        

            """
            [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]

            after union

            [a12,b12,c12,d12,e12] 
            """
            all_subquery_center_encodings = torch.stack(sub_query_center_embedding_list, dim=0)
            all_subquery_offset_encodings = torch.stack(sub_query_offset_embedding_list, dim=0)

            new_query_center_embeddings = self.center_union_net(all_subquery_center_encodings)
            new_query_offset_embeddings = self.offset_union_net(all_subquery_offset_encodings)

            new_query_box_embeddings = tuple([new_query_center_embeddings, new_query_offset_embeddings])
            return new_query_box_embeddings
        
        else:
            all_subquery_encodings = torch.stack(sub_query_box_embedding_list, dim=1)
            batch_size, num_sets, embedding_size = all_subquery_encodings.shape

            flatten_particles = all_subquery_encodings.view(batch_size, -1, embedding_size)


            flatten_particles = self.numerical_union_attn(self.dropout(flatten_particles))
            flatten_particles = self.numerical_union_layer_norm(flatten_particles)
            flatten_particles = self.numerical_union_ffn(flatten_particles) + flatten_particles
            flatten_particles = self.numerical_union_layer_norm(flatten_particles)

            encoding = flatten_particles.sum(dim=1)
            
            return encoding

   

    def loss_fnt(self, sub_query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(sub_query_encoding)

        # [batch_size]
        labels = torch.tensor(labels).type(torch.LongTensor)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    
    def loss_fnt_value(self, query_encoding, labels):
        """
        :param query_encoding: [batch_size, embedding_size]
        :param labels: [batch_size]
        :return:
        """

        # print("query_encoding", query_encoding.shape)
        # print("labels", labels)

        if self.number_encoder_list is not None and len(self.number_encoder_list) > 1:

            label_ids = []

            for label in labels:

                label_offset = self.number_encoder_offsets[label[1]]
              
                label_id = label_offset +  self.number_encoder_list[label[1]].value2id(label[0])

                label_ids.append(label_id)
            
            label_ids = torch.tensor(label_ids)
            label_ids = label_ids.to(self.entity_embedding.weight.device)

            typed_value_embeddings = []
            for encoder_id, encoder in  enumerate(self.number_encoder_list):
                value_embedding = encoder.get_embeddings()
                encoder_id = torch.tensor([encoder_id]).to(self.number_type_embedding.weight.device)
                value_type_embedding = self.number_type_embedding(encoder_id)
                
                typed_value_embeddings.append(value_embedding + value_type_embedding)

            all_value_embeddings = torch.cat(typed_value_embeddings, dim=0)
           

            if self.mixed_value_reprerentation:
                learned_number_encoding_scores = self.decoder(query_encoding)[:, self.num_entities:]
            
             
            query_encoding_scores = torch.matmul(query_encoding, all_value_embeddings.t()) 

            if self.mixed_value_reprerentation:
                query_encoding_scores = query_encoding_scores + learned_number_encoding_scores
            

            
            
            loss = self.numerical_label_smoothing_loss(query_encoding_scores, label_ids)

            return loss
        
        else:

            label_ids = self.number_encoder_list[0].values2ids([l[0] for l in labels])
         
            label_ids = torch.tensor(label_ids)
            label_ids = label_ids.to(self.entity_embedding.weight.device)

            all_value_embeddings = self.number_encoder_list[0].get_embeddings()


            if self.mixed_value_reprerentation:
                learned_number_encoding_scores = self.decoder(query_encoding)[:, self.num_entities:]
                
            query_encoding_scores = torch.matmul(query_encoding, all_value_embeddings.t()) 

            if self.mixed_value_reprerentation:
                query_encoding_scores = query_encoding_scores + learned_number_encoding_scores
            
            
            loss = self.numerical_label_smoothing_loss(query_encoding_scores, label_ids)
            

            return loss 
    


    def forward(self, batched_structured_query, label=None):
        # We need to override this forward function as the structure of box embedding is different
        # input: batched_structured_query
        # output: BOX TUPLE instead of single embedding

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

            
            # intersection of box embedding takes tuple of two lists
            numerical_intersection = False
            sub_query_center_result_list = []
            sub_query_offset_result_list = []
            for _i in range(1, len(batched_structured_query)):
                
                sub_result =  self.forward(batched_structured_query[_i])
                
                if isinstance(sub_result, tuple):
                    sub_query_center_result_list.append(sub_result[0])
                    sub_query_offset_result_list.append(sub_result[1])
                else:
                    sub_query_center_result_list.append(sub_result)
                    numerical_intersection = True
                    
            if numerical_intersection:
     
                this_query_result = self.intersection(sub_query_center_result_list)
            else:

                sub_query_box_result_list = tuple([sub_query_center_result_list, sub_query_offset_result_list])
                this_query_result = self.intersection(sub_query_box_result_list)


        elif batched_structured_query[0] == "u":
            # union of box embedding takes tuple of two lists
            sub_query_center_result_list = []
            sub_query_offset_result_list = []

            numerical_intersection = False
            for _i in range(1, len(batched_structured_query)):
                sub_result =  self.forward(batched_structured_query[_i])
                
                if isinstance(sub_result, tuple):
                    sub_query_center_result_list.append(sub_result[0])
                    sub_query_offset_result_list.append(sub_result[1])
                else:
                    sub_query_center_result_list.append(sub_result)
                    numerical_intersection = True
            

            if numerical_intersection:
     
                this_query_result = self.union(sub_query_center_result_list)

            else:
                sub_query_box_result_list = tuple([sub_query_center_result_list, sub_query_offset_result_list])
                this_query_result = self.union(sub_query_box_result_list)

        elif batched_structured_query[0] == "e":
            # set the offset tensor to all zeros
            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_center_result = self.entity_embedding(entity_ids)
            this_query_offset_result = torch.zeros(this_query_center_result.shape).to(
                self.entity_embedding.weight.device)
            this_query_result = tuple([this_query_center_result, this_query_offset_result])
        

        elif batched_structured_query[0] == "nv":
            
            if len(self.number_encoder_list) > 1:
                
                entity_values = [ self.number_encoder_list[batched_structured_query[1][i]] (batched_structured_query[2][i]) for i in range(len(batched_structured_query[1]))]
                entity_values = torch.cat(entity_values, dim=0)

                type_ids =  torch.tensor(batched_structured_query[1])
                type_ids = type_ids.to(self.number_type_embedding.weight.device)
                type_embeddings = self.number_type_embedding(type_ids)

            

                this_query_result = entity_values + type_embeddings


                label_ids = []

                for i in range(len(batched_structured_query[2])):
                    label_offset = self.number_encoder_offsets[batched_structured_query[1][i]]
                    label_id = label_offset +  self.number_encoder_list[batched_structured_query[1][i]].value2id(batched_structured_query[2][i])
                    label_ids.append(label_id)
                
                label_ids = torch.tensor(label_ids) + self.num_entities
                label_ids = label_ids.to(self.entity_embedding.weight.device)
                learned_embeddings = self.entity_embedding(label_ids)

                this_query_result = this_query_result + learned_embeddings

            
            else:

               
                this_query_result = self.number_encoder_list[0](batched_structured_query[2].tolist())

                label_ids = self.number_encoder_list[0].values2ids(batched_structured_query[2])

                label_ids = torch.tensor(label_ids) + self.num_entities
                label_ids = label_ids.to(self.entity_embedding.weight.device)

                learned_embeddings = self.entity_embedding(label_ids)

                this_query_result = this_query_result + learned_embeddings


        


        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
    
            if isinstance(label[0], int) or isinstance(label[0], np.int64) or isinstance(label[0], np.int32):
                # print("label is int")
                # print   (label)
                loss = self.loss_fnt(this_query_result,  label)

            else:
                # print("label is value")
                
                loss = self.loss_fnt_value(this_query_result, label)
            return loss

def unit_test_baseline():
    
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

    model = Q2B(num_entities=nentity + nvalue,
                    num_relations=nrelation + nattribute * 2 + nnumerical_proj,
                    embedding_size=300)
    if torch.cuda.is_available():
        model = model.cuda()




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

            print(batched_query)
        
            query_embedding = model(batched_query)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

            

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
        
            query_embedding = model(batched_query)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    

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

        query_embedding = model(batched_query)
        # print(query_embedding.shape)
        loss = model(batched_query, positive_sample)
        print(loss)

def unit_test():
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
        




    train_typed_values = {}
    for u in train_graph.nodes():
        if isinstance(u, tuple):
            if u[1] not in train_typed_values:
                train_typed_values[u[1]] = []
            train_typed_values[u[1]].append(u[0])
    

    encoder_list  = []

    for i in range(len(all_typed_values)):
        # positional_encoder_log = PositionalEncoder(output_size=300, train_values=train_typed_values[i], all_values=all_typed_values[i], n=10000)
        # encoder_list.append(positional_encoder_log)


        positional_encoder_log = DICE(output_size=300, train_values=train_typed_values[i], all_values=all_typed_values[i])
        encoder_list.append(positional_encoder_log)


    # positional_encoder_log = PositionalEncoder(output_size=300, train_values=train_values, all_values=all_values, n=10000)
    # dice_encoder_log = DICE(output_size=300, train_values=train_values, all_values=all_values)

    # positional_encoder_quantile = PositionalEncoder(output_size=300, train_values=train_values, all_values=all_values, n=10000, scaler="quantile")
    # dice_encoder_quantile = DICE(output_size=300, train_values=train_values, all_values=all_values, scaler="quantile")

    # lstm_encoder = DigitRNN(output_size=300, train_values=train_values, all_values=all_values)
    # gmm_encoder = GMM_Prototype(output_size=300, train_values=train_values, all_values=all_values)

    

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

    nrelation = max(relation_edges_list) + 10
    nattribute = max(attribute_edges_list) + 10

    nnumerical_proj = max(numerical_edges_list) + 10
    
    q2p_model = Q2B(num_entities=nentity,
                    num_relations=nrelation,
                    embedding_size=300,
                    num_attributes=nattribute,
                    num_numrical_proj=nnumerical_proj,
                    value_vocab=value_vocab,
                    number_encoder_list=encoder_list,
                    mixed_value_reprerentation=True,
                    )
    if torch.cuda.is_available():
        q2p_model = q2p_model.cuda()

    batch_size = 5

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
        # print(batched_query)
        # print(unified_ids)
        # print(positive_sample)

        query_embedding = q2p_model(batched_query)
        # print(query_embedding.shape)
        loss = q2p_model(batched_query, positive_sample)
        print(loss)

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

    for key, loader in baseline_validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            # print(unified_ids)
            # print([len(_) for _ in train_answers])
            # print([len(_) for _ in valid_answers])

            query_embedding = q2p_model(batched_query)
            result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = q2p_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

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

    for key, loader in baseline_test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            # print(unified_ids)
            # print(train_answers[0])
            # print([len(_) for _ in train_answers])
            # print([len(_) for _ in valid_answers])
            # print([len(_) for _ in test_answers])

            query_embedding = q2p_model(batched_query)
            result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = q2p_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)
            break
    

def unit_test_single_encoder():

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


    all_values = []
    for u in test_graph.nodes():
        if isinstance(u, tuple):
            all_values.append(u[0])

    train_values = []
    for u in train_graph.nodes():
        if isinstance(u, tuple):
            train_values.append(u[0])


    all_typed_values = {}


    train_typed_values = {}


    positional_encoder_log = PositionalEncoder(output_size=300, train_values=train_values, all_values=all_values, n=10000)
    dice_encoder_log = DICE(output_size=300, train_values=train_values, all_values=all_values)

    positional_encoder_quantile = PositionalEncoder(output_size=300, train_values=train_values, all_values=all_values, n=10000, scaler="quantile")
    dice_encoder_quantile = DICE(output_size=300, train_values=train_values, all_values=all_values, scaler="quantile")

    lstm_encoder = DigitRNN(output_size=300, train_values=train_values, all_values=all_values)
    gmm_encoder = GMM_Prototype(output_size=300, train_values=train_values, all_values=all_values)

    

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

    nrelation = max(relation_edges_list) + 10
    nattribute = max(attribute_edges_list)+ 10

    nnumerical_proj = max(numerical_edges_list) + 10
    for numeral_encoder in [positional_encoder_log, dice_encoder_log]:
        q2p_model = Q2B(num_entities=nentity,
                        num_relations=nrelation,
                        embedding_size=300,
                        num_attributes=nattribute,
                        num_numrical_proj=nnumerical_proj,
                        value_vocab=value_vocab,
                        number_encoder_list=[numeral_encoder],
                        mixed_value_reprerentation=True,
                        )
        if torch.cuda.is_available():
            q2p_model = q2p_model.cuda()

        batch_size = 5

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
            # print(batched_query)
            # print(unified_ids)
            # print(positive_sample)

            query_embedding = q2p_model(batched_query)
            # print(query_embedding.shape)
            loss = q2p_model(batched_query, positive_sample)
            print(loss)

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

        for key, loader in baseline_validation_loaders.items():
            print("read ", key)
            for batched_query, unified_ids, train_answers, valid_answers in loader:
                print(batched_query)
               

                query_embedding = q2p_model(batched_query)
                result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
                print(result_logs)

                result_logs = q2p_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
                print(result_logs)

                break

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

        for key, loader in baseline_test_loaders.items():
            print("read ", key)
            for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
                print(batched_query)
                
                query_embedding = q2p_model(batched_query)
                result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
                print(result_logs)

                result_logs = q2p_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
                print(result_logs)
                break



if __name__ == "__main__":

    unit_test_baseline()

    unit_test()

    unit_test_single_encoder()


    
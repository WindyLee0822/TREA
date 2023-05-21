from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import numpy as np
import json
device='cuda'
def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings
type_names = ["Candidate", "Movie", "Actor", "Director", "Genre", "Time", "Attr", "Subject", "None"]
relations_names = ['time', 'director', 'starring', 'genre', 'subject', 'belong', 'timeR', 'directorR', 'starringR','genreR', 'subjectR', 'belongR']

EDGE_TYPES = [58, 172]
def _edge_list(graph):

    nodes = graph['nodes']

    relations = graph['relations']

    edge_index = [[], []]
    edge_type = []

    num_nodes = len(nodes)
    node_feature = torch.zeros(num_nodes, 9)
    for i, node in enumerate(nodes):
        if node['type'] == "Person":
            for item in node['role']:
                type_idx = type_names.index(item)
                node_feature[i][type_idx] = 1
        elif node['type'] == "Attr" and node['name'] == "None":
            type_idx = type_names.index("None")
            node_feature[i][type_idx] = 1
        else:
            type_idx = type_names.index(node['type'])
            node_feature[i][type_idx] = 1

    for relation in relations:
        if relation[0]>=30452 and relation[0]<=30458 or relation[1]>=30452 and relation[1]<=30458:
            continue
        edge_index[0].append(int(relation[0]))
        edge_index[1].append(int(relation[1]))
        edge_type.append(relations_names.index(relation[2]))

    edge_index = torch.from_numpy(np.array(edge_index)).to(device)
    edge_type = torch.from_numpy(np.array(edge_type)).long().to(device)
    return edge_index,edge_type,len(relations_names)


# def _edge_list(kg, n_entity, hop):
#     edge_list = []
#     for h in range(hop):
#         for entity in range(n_entity):
#             # add self loop
#             # edge_list.append((entity, entity))
#             # self_loop id = 185
#             edge_list.append((entity, entity, 185))
#             if entity not in kg:
#                 continue
#             for tail_and_relation in kg[entity]:
#                 if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
#                     edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
#                     edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))
#
#     relation_cnt = defaultdict(int)
#     relation_idx = {}
#     for h, t, r in edge_list:
#         relation_cnt[r] += 1
#     for h, t, r in edge_list:
#         if relation_cnt[r] > 1000 and r not in relation_idx:
#             relation_idx[r] = len(relation_idx)
#
#     return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def concept_edge_list4GCN():
    node2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
    f=open('conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).to(device)

def concept_egde_list4GCN_new():
    node2index = json.load(open('key2index_3rd.json', encoding='utf-8'))
    f = open('conceptnet_edges2nd.txt', encoding='utf-8')
    edges = [[],[]]
    r=[]
    stopwords = set([word.strip() for word in open('stopwords.txt', encoding='utf-8')])
    for line in f:
        lines = line.strip().split('\t')
        entity0 = node2index[lines[1].split('/')[0]]
        entity1 = node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges[0].append(entity0)
        edges[1].append(entity1)
        if lines[0] in ['NotHasProperty','NotDesires','Antonym','NotCapableOf','DistinctFrom']:
            r.append(1)
            edges[0].append(entity1)
            edges[1].append(entity0)
            r.append(1)
        else:
            r.append(0)
            edges[0].append(entity1)
            edges[1].append(entity0)
            r.append(0)

    edge_index = torch.from_numpy(np.array(edges)).to(device)
    edge_type = torch.from_numpy(np.array(r)).long().to(device)
    return edge_index,edge_type
def concept_egde_list4GCN_newnew():
    concept_edge_kind=['CreatedBy', 'HasPrerequisite', 'SimilarTo', 'DerivedFrom', 'MotivatedByGoal', 'FormOf', 'CausesDesire', 'NotDesires', 'LocatedNear', 'HasProperty', 'EtymologicallyRelatedTo', 'UsedFor', 'HasFirstSubevent', 'dbpedia/occupation', 'NotCapableOf', 'dbpedia/knownFor', 'MannerOf', 'CapableOf', 'DistinctFrom', 'HasLastSubevent', 'Desires', 'Antonym', 'dbpedia/genre', 'dbpedia/language', 'RelatedTo', 'Entails', 'MadeOf', 'EtymologicallyDerivedFrom', 'HasSubevent', 'AtLocation', 'HasA', 'dbpedia/product', 'Causes', 'dbpedia/genus', 'dbpedia/influencedBy', 'IsA', 'InstanceOf', 'dbpedia/capital', 'Synonym', 'ReceivesAction', 'dbpedia/field', 'NotHasProperty', 'PartOf', 'HasContext', 'DefinedAs']
    node2index = json.load(open('key2index_3rd.json', encoding='utf-8'))
    f = open('conceptnet_edges2nd.txt', encoding='utf-8')
    edges = [[],[]]
    r=[]
    kind_num=json.load(open('kind_num.json'))
    kind=[k for k,v in kind_num.items() if v<500]
    concept_edge_kind=[i for i in concept_edge_kind if i not in kind]
    stopwords = set([word.strip() for word in open('stopwords.txt', encoding='utf-8')])
    for line in f:
        lines = line.strip().split('\t')
        entity0 = node2index[lines[1].split('/')[0]]
        entity1 = node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        if lines[0] in kind:
            continue
        e_kind=concept_edge_kind.index(lines[0])
        edges[0].append(entity0)
        edges[1].append(entity1)
        r.append(e_kind)
        edges[0].append(entity1)
        edges[1].append(entity0)
        r.append(e_kind)
        # if lines[0] in ['NotHasProperty','NotDesires','Antonym','NotCapableOf','DistinctFrom']:
        #     r.append(1)
        #     edges[0].append(entity1)
        #     edges[1].append(entity0)
        #     r.append(1)
        # else:
        #     r.append(0)
        #     edges[0].append(entity1)
        #     edges[1].append(entity0)
        #     r.append(0)

    edge_index = torch.from_numpy(np.array(edges)).to(device)
    edge_type = torch.from_numpy(np.array(r)).long().to(device)
    return edge_index,edge_type,len(concept_edge_kind)
class GateLayer_3_eles(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer_3_eles, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 3, input_dim)
        self.act = nn.LeakyReLU()
        self._norm_layer2 = nn.Linear(input_dim, 3)

    def forward(self, input1, input2, input3):
        norm_input = self._norm_layer1(torch.cat([input1, input2, input3], dim=-1))
        norm_input = self.act(norm_input)
        gate = F.softmax(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate[:, 0].view(-1,1) * input1 + gate[:, 1].view(-1,1) * input2 + gate[:, 2].view(-1,1) * input3 #(bs, dim)
        return gated_emb
class GateLayer(nn.Module):
    def __init__(self, input_dim,is_minus=False):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)
        self.minus=is_minus

    def forward(self, input1, input2):
        if self.minus:
            norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
            gate = torch.sigmoid(self._norm_layer2(norm_input))*0.2  # (bs, 1)
            gated_emb = input1 - gate * input2
        else:
            norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
            gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
            gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb


class gen_CrossModel(nn.Module):
    def __init__(self, opt, dictionary,con2word,n_kind_set, is_finetune='mov', padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.classify=False
        self.device='cuda'
        self.dim=opt['dim']
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']
        self.entity_max=opt['n_entity']
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label
        self.entity_label=[6] * 6924 + [5] * 12803 + [4] * 10707 + [1] * 18 + [0] * 7 + [2] * 12

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg=json.load(open('data/redial_kg.json'))

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion1=nn.BCELoss(reduce=False)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.criterion2 = nn.CrossEntropyLoss(reduce=False,ignore_index=self.entity_max)
        self.criterion3=nn.CrossEntropyLoss(reduce=False,ignore_index=2)

        self.self_attn1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_emb1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_emb2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        # self.self_attn4gen = SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        # self.self_attn_db4gen = SelfAttentionLayer(opt['dim'],opt['dim'])
        self.user_norm4gen=nn.Linear(opt['dim']*3,opt['dim'])
        self.output_en4gen=nn.Linear(opt['dim'],opt['n_entity'])

        self.self_attn_batch_db = SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.self_attn_batch_his1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_his2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_hism1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_hism2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_leaf = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])
        self.self_attn_word=SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.semantic_drag=nn.Sequential(
            nn.Linear(opt['embedding_size'],opt['dim']),
            nn.Tanh(),
            nn.Linear(opt['dim'],opt['embedding_size'])
        )
        self.semantic_drag_new = nn.Sequential(
            nn.Linear(opt['embedding_size'], opt['dim']),
            nn.Tanh(),
            nn.Linear(opt['dim'], opt['dim'])
        )
        self.self_attn_word_new = SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.self_conattn=SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Sequential(nn.Linear(opt['dim'], opt['dim']),nn.LeakyReLU())
        self.info_db_norm = nn.Sequential(nn.Linear(opt['dim'], opt['dim']),nn.LeakyReLU())
        self.state_norm1 = nn.Sequential(nn.Linear(opt['dim']*24,opt['dim']*12),nn.LeakyReLU(),nn.Linear(opt['dim']*12,opt['dim']*24))
        self.state_norm2 = nn.Linear(opt['dim']*24, opt['dim'])
        self.con_word_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_output_word=nn.Linear(opt['embedding_size'],self.embeddings.weight.shape[0])
        self.dim2dimSe=nn.Sequential(nn.Linear(self.dim,self.dim),nn.LeakyReLU())
        self.dim2dim = nn.Linear(self.dim,self.dim)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)
        self.score_loss = nn.MSELoss(size_average=False, reduce=False)

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)
        self.gate_norm_3_ele = GateLayer_3_eles(self.dim)
        self.gate_layer1 = GateLayer(self.dim)
        self.gate_layer2 = GateLayer(self.dim)
        self.gate_layer3 = GateLayer(self.dim)
        self.gate_layer_minus = GateLayer(self.dim,is_minus = True)
        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        # edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        # edge_list = list(set(edge_list))
        # print(len(edge_list), self.n_relation)
        # self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda()
        # self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        # self.db_edge_type = self.dbpedia_edge_sets[:, 2]
        self.db_edge_idx,self.db_edge_type,self.n_relation=_edge_list(self.kg)
        # self.con_edge_idx,self.con_edge_type=concept_egde_list4GCN_new()
        self.con_edge_idx, self.con_edge_type,self.con_kind = concept_egde_list4GCN_newnew()
        self.con_re_emb = nn.Embedding(self.con_kind, self.dim)
        self.concept_relation=2
        self.con_classify = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.concept_relation)
        )

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation+1, num_bases=opt['num_bases'])
        # self.db_nodes_features = nn.Embedding(opt['n_entity'],self.dim)
        # self.con_nodes_features = nn.Embedding(opt['n_concept'] + 1, self.dim)
        # nn.init.xavier_normal_(self.db_nodes_features.weight)
        # nn.init.xavier_normal_(self.con_nodes_features.weight)
        self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.concept_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)
        self.n_kind=6
        self.kind_emb=nn.Embedding(self.n_kind,self.embedding_size)
        self.wordfusekind=nn.Linear(self.embedding_size*2,self.embedding_size)

        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}
        self.con2word=torch.tensor(con2word,device=self.device)
        self.n_kind_set=n_kind_set
        self.stopwords_rd=torch.tensor([dictionary[word.strip()] for word in open('stopwords.txt', encoding='utf-8') if word in dictionary],device=self.device).long()
        # self.ENTTOKEN_rd=n_kind_set[0]
        # self.MOVIE_rd=n_kind_set[1]
        # self.classlinear1=nn.Linear(self.embedding_size*2,5)
        self.emb2ent=nn.Sequential(
            nn.Linear(self.embedding_size,self.embedding_size//2),
            nn.ReLU(),
            nn.Linear(self.embedding_size//2,self.embedding_size)
        )
        self.ent2rec=nn.Linear(2*self.dim+self.embedding_size,self.dim)
        # self.mask4key=torch.Tensor(np.load('mask4key.npy')).cuda()
        # self.mask4movie=torch.Tensor(np.load('mask4movie.npy')).cuda()
        self.mask4=torch.ones(len(dictionary) + 4).to(self.device)
        # if is_finetune=='mov':
        #     params=[self.embeddings.parameters()]
        #     for param in params:
        #         for pa in param:
        #             pa.requires_grad = False

        params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),self.concept_embeddings.parameters()]
        # self.con_classify.parameters(),
        # self.concept_embeddings.parameters(),self.kind_emb.parameters(),
        # self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
        # self.gate_norm.parameters(), self.output_en.parameters(),
        # self.user_norm4gen.parameters(),self.output_en4gen.parameters()]
        for param in params:
            for pa in param:
                pa.requires_grad = False

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    # def decode_greedy(self, encoder_states,kind_logit, bsz, maxlen,db_user_emb,con_user_emb,kg_encoding):
    #     """
    #     Greedy search
    #
    #     :param int bsz:
    #         Batch size. Because encoder_states is model-specific, it cannot
    #         infer this automatically.
    #
    #     :param encoder_states:
    #         Output of the encoder model.
    #
    #     :type encoder_states:
    #         Model specific
    #
    #     :param int maxlen:
    #         Maximum decoding length
    #
    #     :return:
    #         pair (logits, choices) of the greedy decode
    #
    #     :rtype:
    #         (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
    #     """
    #     xs = self._starts(bsz)
    #     seqlen=xs.shape[1]
    #     special_token_mask=[] #torch.zeros(xs.shape[0],xs.shape[1],device=self.device)
    #
    #     incr_state = None
    #     logits_ = []
    #     latent=[]
    #     # classify_score=[]
    #
    #     for i in range(maxlen):
    #         # if self.classify==False:
    #         #
    #         #     special_token_mask = torch.zeros_like(xs, device=self.device)
    #         #     special_token_emb = torch.zeros(xs.shape[0], xs.shape[1], self.embedding_size, device=self.device)
    #         #     for ii, token in enumerate(self.n_kind_set.values()):
    #         #         special_token_mask = torch.where(xs == token, special_token_mask, ii)
    #         #         # print(inputs.shape,special_token_emb.shape,word_logit[:,i].unsqueeze(1).repeat(1,seqlen,1).shape)
    #         #         special_token_emb = torch.where(xs.unsqueeze(-1).repeat(1, 1, self.embedding_size) == token,
    #         #                                         special_token_emb, kind_logit[:,ii].unsqueeze(1).repeat(1,xs.shape[1], 1))
    #         # # elif self.classify==True:
    #         # #     movie_pos=torch.where(xs[:,-1]==self.MOVIE_rd,6,0)
    #         # #     if logits_!=[] and (xs[:,-1]==self.ENTTOKEN_rd).sum()!=0:
    #         # #         # class_latent = torch.gather(latent, dim=1,index=special_mask_sparse.indices()[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.embedding_size)).squeeze(1)
    #         # #         class_info=torch.cat([latent,user_emb],dim=-1)
    #         # #         class_score=F.softmax(self.classlinear1(class_info),dim=-1)
    #         # #         class_res=class_score*special_max_score
    #         # #         classify_score.append(class_score)
    #         # #
    #         # #     else:
    #         # #         class_res=torch.zeros(xs.shape[0],5,device=self.device)
    #         # #         classify_score.append(class_res)
    #         # #
    #         # #     if class_res.sum()!=0:
    #         # #         class_final_=class_res.max(-1).indices
    #         # #         class_final=torch.where(movie_pos==6,6,class_final_)
    #         # #         if special_token_mask!=[]:
    #         # #             special_token_mask=torch.cat([special_token_mask,class_final.T+1],dim=-1)
    #         # #         else:
    #         # #             special_token_mask=torch.zeros(xs.shape[0], xs.shape[1], device=self.device)
    #         # #
    #         # #     else:
    #         # #         if special_token_mask!=[]:
    #         # #             # print(special_token_mask.shape,movie_pos.shape)
    #         # #             special_token_mask=torch.cat([special_token_mask,movie_pos.unsqueeze(0).T],dim=-1)
    #         # #         else:
    #         # #             special_token_mask=torch.zeros(xs.shape[0], xs.shape[1], device=self.device)
    #         # #     special_token_emb = torch.zeros(xs.shape[0], xs.shape[1], self.embedding_size, device=self.device)
    #         # #
    #         # #     for i in range(6):
    #         # #         special_token_emb = torch.where(special_token_mask.unsqueeze(-1).repeat(1, 1, self.embedding_size) == i,
    #         # #                                         special_token_emb, kind_logit[:, i].unsqueeze(1).repeat(1, seqlen, 1))
    #         # #
    #         # #     # special_token_mask = torch.cat([0, special_token_mask], -1)[:-1]
    #         # #     # special_token_emb = torch.cat([torch.zeros(bsz, 1, self.embedding_size), special_token_emb], -2)[:, :-1, :]
    #         #
    #         # # todo, break early if all beams saw EOS
    #         scores, incr_state = self.decoder(xs, encoder_states,kind_logit,kg_encoding)
    #         #batch*1*hidden
    #         latent_= scores[:, -1:, :]
    #         #scores = self.output(scores)
    #         # kg_attn_norm = self.kg_attn_norm(attention_kg)
    #         #
    #         # db_attn_norm = self.db_attn_norm(attention_db)
    #         #
    #         # copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))
    #         #
    #         # # logits = self.output(latent)
    #         # con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
    #         # voc_logits = F.linear(scores, self.embeddings.weight)
    #         # # print(logits.size())
    #         # # print(mem_logits.size())
    #         # #gate = F.sigmoid(self.gen_gate_norm(scores))
    #
    #         # sum_logits = voc_logits + con_logits #* (1 - gate)
    #         '''下面注释了'''
    #         # kg_attention_latent = self.kg_attn_norm(con_user_emb)
    #         # db_attention_latent = self.db_attn_norm(db_user_emb)
    #         # copy_latent = self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
    #         #                                         db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1), latent_],
    #         #                                        -1))
    #         #
    #         #
    #         # con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)  # F.linear(copy_latent, self.embeddings.weight)
    #         logits = F.linear(latent_, self.embeddings.weight)
    #         sum_logits = logits #+ con_logits  # *(1-gate)
    #         _, preds = sum_logits.max(dim=-1)
    #         #scores = F.linear(scores, self.embeddings.weight)
    #
    #         #print(attention_map)
    #         #print(db_attention_map)
    #         #print(preds.size())
    #         #print(con_logits.size())
    #         #exit()
    #         #print(con_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
    #         #print(voc_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
    #
    #         #print(torch.topk(voc_logits.squeeze(0).squeeze(0),k=50)[1])
    #
    #         #sum_logits = scores
    #         # print(sum_logits.size())
    #
    #         #_, preds = sum_logits.max(dim=-1)
    #         logits_.append(sum_logits)
    #         xs = torch.cat([xs, preds], dim=1)
    #         # check if everyone has generated an end token
    #         all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
    #         if all_finished:
    #             break
    #     logits_ = torch.cat(logits_, 1)
    #     # classify_score=torch.stack(classify_score).transpose(0,1)#seq,bsz,6->bsz,sql,6
    #     return logits_, xs
    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz,
                      maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            # batch*1*hidden
            scores = scores[:, -1:, :]
            # scores = self.output(scores)
            # kg_attn_norm = self.kg_attn_norm(attention_kg)
            #
            # db_attn_norm = self.db_attn_norm(attention_db)
            #
            # copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))
            #
            # # logits = self.output(latent)
            # con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(
            #     0)  # F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            # print(logits.size())
            # print(mem_logits.size())
            # gate = F.sigmoid(self.gen_gate_norm(scores))

            sum_logits = voc_logits #+ con_logits  # * (1 - gate)
            _, preds = sum_logits.max(dim=-1)
            # scores = F.linear(scores, self.embeddings.weight)

            # print(attention_map)
            # print(db_attention_map)
            # print(preds.size())
            # print(con_logits.size())
            # exit()
            # print(con_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            # print(voc_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])

            # print(torch.topk(voc_logits.squeeze(0).squeeze(0),k=50)[1])

            # sum_logits = scores
            # print(sum_logits.size())

            # _, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden

        # kg_attention_latent=self.kg_attn_norm(attention_kg)
        # db_attention_latent=self.db_attn_norm(attention_db)
        # copy_latent=self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))
        #
        # con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)

        sum_logits = logits #+ con_logits
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    # def decode_forced_new(self, encoder_states, ys, word_logit,special_mask,kg_encoding,db_encoding):
    #     bsz = ys.size(0)
    #     seqlen = ys.size(1)
    #     inputs = ys.narrow(1, 0, seqlen - 1)
    #     inputs = torch.cat([self._starts(bsz), inputs], 1)
    #     special_token_mask=torch.cat([torch.zeros(special_mask.shape[0],1,device=self.device),special_mask[:,1:]],dim=-1)
    #
    #     special_token_emb=torch.zeros(bsz,seqlen,self.embedding_size,device=self.device)
    #
    #     #todo 这两行开始没删
    #     # kind_logit_=self.wordfusekind(torch.cat([word_logit,self.kind_emb.weight.unsqueeze(0).repeat(bsz,1,1)],dim=-1))#bsz,n_kind,emb
    #     # kind_logit=kind_logit_+word_logit+self.kind_emb.weight.unsqueeze(0).repeat(bsz,1,1)
    #
    #     # special_token_mask=torch.zeros_like(inputs,device=self.device)
    #     # for i,token in enumerate(self.n_kind_set.values()):
    #     #     special_token_mask=torch.where(inputs==token,special_token_mask,i)
    #     #     # print(inputs.shape,special_token_emb.shape,word_logit[:,i].unsqueeze(1).repeat(1,seqlen,1).shape)
    #     #     special_token_emb=torch.where(inputs.unsqueeze(-1).repeat(1,1,self.embedding_size)==token,special_token_emb,kind_logit[:,i].unsqueeze(1).repeat(1,seqlen,1))
    #     for i in range(6):
    #         special_token_emb=torch.where(special_token_mask.unsqueeze(-1).repeat(1,1,self.embedding_size)==i,special_token_emb,word_logit[:,i].unsqueeze(1).repeat(1,seqlen,1))
    #     # special_token_mask=torch.cat([0,special_token_mask],-1)[:-1]
    #     # special_token_emb=torch.cat([torch.zeros(bsz,1,self.embedding_size),special_token_emb],-2)[:,:-1,:]
    #
    #     latent, _ = self.decoder(inputs,encoder_states,special_token_mask,special_token_emb,kg_encoding,db_encoding)  # batch*r_l*hidden
    #     logits = F.linear(latent, self.embeddings.weight)
    #
    #     _, preds = logits.max(dim=2)
    #
    #     return logits,preds,latent
    def decode_forced_new(self, encoder_states, ys, word_logit,db_user_emb,con_user_emb,kg_encoding):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        # special_token_mask=torch.cat([torch.zeros(special_mask.shape[0],1,device=self.device),special_mask[:,1:]],dim=-1)
        #
        # special_token_emb=torch.zeros(bsz,seqlen,self.embedding_size,device=self.device)
        #
        # #todo 这两行开始没删
        # # kind_logit_=self.wordfusekind(torch.cat([word_logit,self.kind_emb.weight.unsqueeze(0).repeat(bsz,1,1)],dim=-1))#bsz,n_kind,emb
        # # kind_logit=kind_logit_+word_logit+self.kind_emb.weight.unsqueeze(0).repeat(bsz,1,1)
        #
        # # special_token_mask=torch.zeros_like(inputs,device=self.device)
        # # for i,token in enumerate(self.n_kind_set.values()):
        # #     special_token_mask=torch.where(inputs==token,special_token_mask,i)
        # #     # print(inputs.shape,special_token_emb.shape,word_logit[:,i].unsqueeze(1).repeat(1,seqlen,1).shape)
        # #     special_token_emb=torch.where(inputs.unsqueeze(-1).repeat(1,1,self.embedding_size)==token,special_token_emb,kind_logit[:,i].unsqueeze(1).repeat(1,seqlen,1))
        # for i in range(6):
        #     special_token_emb=torch.where(special_token_mask.unsqueeze(-1).repeat(1,1,self.embedding_size)==i,special_token_emb,word_logit[:,i].unsqueeze(1).repeat(1,seqlen,1))
        # # special_token_mask=torch.cat([0,special_token_mask],-1)[:-1]
        # # special_token_emb=torch.cat([torch.zeros(bsz,1,self.embedding_size),special_token_emb],-2)[:,:-1,:]

        latent_, _ = self.decoder(inputs,encoder_states,word_logit,kg_encoding)  # batch*r_l*hidden

        # kg_attention_latent = self.kg_attn_norm(con_user_emb)
        # db_attention_latent = self.db_attn_norm(db_user_emb)
        # copy_latent = self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
        #                                         db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1), latent_], -1))
        #
        #
        # con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)  # F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent_, self.embeddings.weight)

        sum_logits = logits #+ con_logits  # *(1-gate)
        _, preds = sum_logits.max(dim=2)

        return sum_logits,preds,latent_

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, con_label, db_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)#bsz,dim
        db_emb=self.info_db_norm(db_user_emb)#bsz,dim

        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        #  word_set[:,0:4]=0
        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.to(self.device).float()),dim=-1)*mask.to(self.device)\
                        # +torch.sum(self.info_con_loss(word_con_score.float(),word_set.float()),dim=-1)

                  # +torch.sum(self.info_db_loss(related_db_scores,db_label.to(self.device).float()),dim=-1)*mask.to(self.device)
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.to(self.device).float()),dim=-1)*mask.to(self.device)
        # info_score=F.sigmoid(torch.matmul(con_emb,db_emb.transpose(0,1)))#bsz,dim * bsz,dim-> bsz,bsz
        # info_score_sum=info_score.sum(-1)
        # pos_score=info_score.diagonal()
        # neg_score=(info_score_sum-pos_score)/(con_user_emb.shape[0]-1)
        # info_db_loss=-(pos_score-neg_score)*mask.to(self.device)


        return torch.mean(info_db_loss), torch.mean(info_con_loss)

    def compute_barrow_loss(self, view_1_rep, view_2_rep, mu):
        cov_matrix_up = torch.matmul(view_1_rep, view_2_rep.t())
        bs = view_1_rep.shape[0]
        words_down = (view_1_rep * view_1_rep).sum(dim=1).view(bs, 1)
        entities_down = (view_2_rep * view_2_rep).sum(dim=1).view(1, bs)
        words_down = words_down.expand(bs, bs)
        entities_down = entities_down.expand(bs, bs)
        cov_matrix_down = torch.sqrt(words_down * entities_down + 1e-12)
        cov_matrix = cov_matrix_up / cov_matrix_down
        mask_part1 = torch.eye(bs).to(self.device)
        mask_part2 = torch.ones((bs, bs)).to(self.device) - mask_part1

        loss_part1 = ((mask_part1 - cov_matrix).diag() * (mask_part1 - cov_matrix).diag()).sum()
        loss_part2 = ((mask_part2 * cov_matrix) * (mask_part2 * cov_matrix)).sum()
        loss = mu * loss_part1 + (1-mu) * loss_part2

        return loss_part1, loss_part2, loss
    # def pretrain_infomax(self, history_words_reps, history_entities_reps,user_id):
    #
    #
    #     words_sim_matrix_up = torch.matmul(history_words_reps, history_words_reps.t())
    #     words_dd = words_sim_matrix_up.diag()
    #     words_bs = history_words_reps.shape[0]
    #     words_down1 = torch.sqrt(words_dd.view(-1, 1).repeat(1, words_bs))
    #     words_down2 = torch.sqrt(words_dd.view(1, -1).repeat(words_bs, 1))
    #     words_sim_matrix = words_sim_matrix_up / words_down1 / words_down2
    #     words_mask = (words_sim_matrix > 0.85)
    #     words_sim_matrix = (words_sim_matrix - 0.85 + 0.85 * torch.eye(words_bs).cuda()) * words_mask
    #
    #     entities_sim_matrix_up = torch.matmul(history_entities_reps, history_entities_reps.t())
    #     entities_dd = entities_sim_matrix_up.diag()
    #     entities_bs = history_entities_reps.shape[0]
    #     entities_down1 = torch.sqrt(entities_dd.view(-1, 1).repeat(1, entities_bs))
    #     entities_down2 = torch.sqrt(entities_dd.view(1, -1).repeat(entities_bs, 1))
    #     entities_sim_matrix = entities_sim_matrix_up / entities_down1 / entities_down2
    #     entities_mask = (entities_sim_matrix > 0.85)
    #     entities_sim_matrix = (entities_sim_matrix - 0.85 + 0.85 * torch.eye(entities_bs).cuda()) * entities_mask
    #
    #     w_user_mask_1 = user_id.view(1, -1).repeat(words_bs, 1)
    #     w_user_mask_2 = user_id.view(-1, 1).repeat(1, words_bs)
    #     user_mask = (w_user_mask_1 != w_user_mask_2) + torch.eye(words_bs).cuda()
    #     words_sim_matrix = words_sim_matrix * user_mask
    #
    #     e_user_mask_1 = user_id.view(1, -1).repeat(entities_bs, 1)
    #     e_user_mask_2 = user_id.view(-1, 1).repeat(1, entities_bs)
    #     user_mask = (e_user_mask_1 != e_user_mask_2) + torch.eye(entities_bs).cuda()
    #     entities_sim_matrix = entities_sim_matrix * user_mask
    #
    #     word_attn_rep_weighted = []
    #     for i in range(words_bs):
    #         word_attn_rep_weighted.append((word_attn_rep * words_sim_matrix[i].view(-1, 1)).sum(dim=0))
    #
    #     entity_attn_rep_weighted = []
    #     for i in range(entities_bs):
    #         entity_attn_rep_weighted.append((entity_attn_rep * entities_sim_matrix[i].view(-1, 1)).sum(dim=0))
    #     word_attn_rep_add_lookalike = torch.stack(word_attn_rep_weighted)
    #     entity_attn_rep_add_lookalike = torch.stack(entity_attn_rep_weighted)
    #
    #     word_info_rep = self.word_infomax_norm(word_attn_rep_add_lookalike)
    #     entity_info_rep = self.entity_infomax_norm(entity_attn_rep_add_lookalike)
    #     history_word_rep = self.word_infomax_norm(history_words_reps)
    #     history_entity_rep = self.entity_infomax_norm(history_entities_reps)
    #
    #     his_part1, his_part2, loss_1 = self.compute_barrow_loss(history_word_rep, history_entity_rep, 0.1)
    #     cur_part1, cur_part2, loss_2 = self.compute_barrow_loss(word_info_rep, entity_info_rep, 0.1)
    #
    #     loss = loss_1 + loss_2
    #     # print(pred_item_loss)
    #     return None, loss
    #

    def forward(self, xs, ys,entity, concept, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if test == False:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        #xxs = self.embeddings(xs)
        #mask=xs == self.pad_idx
        bsz = xs.shape[0]
        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)
        # encoder_states,_ = self.encoder(xs)
        # sem_emb,_ = self.self_attn_emb1(encoder_states,xs==self.pad_idx)
        # cur_encoder_states,_ = self.encoder(xs_cur)
        # cur_sem_emb,_ = self.self_attn_emb2(cur_encoder_states,xs_cur==self.pad_idx)

        # token_mask=xs
        # for v in self.n_kind_set.values():
        #     for value in v:
        #         token_mask=(xs!=value)
        # encoder_states=encoder_states[0],encoder_states[1]*token_mask

        # graph network
        # self.db_edge_idx = torch.cat([add_edge_idx,self.db_edge_idx],-1)
        # self.db_edge_type = torch.cat([add_edge_type, self.db_edge_type], -1)
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        db_nodes_features=torch.cat([db_nodes_features,torch.zeros(1,self.dim,device=self.device)],dim=0)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        # '''Here classify each relation of ConceptNet'''
        # # con_re=self.con_classify(self.con_re_emb.weight).argmax(dim=-1)
        # # con_edge_type=con_re[self.con_edge_type]
        # #
        # # con_nodes_features=self.concept_RGCN(None,self.con_edge_idx,con_edge_type)
        #
        # if False:
        #     user_representation_list = []
        #     # db_list4gen=[]
        #     db_con_mask=[]
        #     for i, seed_set in enumerate(seed_sets):
        #         if seed_set == []:
        #             user_representation_list.append(torch.zeros(self.dim).to(self.device))
        #             # db_list4gen.append(torch.zeros(self.dim).to(self.device))
        #             db_con_mask.append(torch.zeros([1]))
        #             continue
        #         user_representation = db_nodes_features[seed_set]  # torch can reflect
        #         user_representation = self.self_attn_db(user_representation)
        #         user_representation_list.append(user_representation)
        #         db_con_mask.append(torch.ones([1]))
        #
        #     db_user_emb=torch.stack(user_representation_list)
        #     # db_user_emb,_=self.self_attn_batch_db(db_nodes_features[entity],entity==self.entity_max)
        #
        #     db_con_mask=torch.stack(db_con_mask)
        #
        #     graph_con_emb=con_nodes_features[concept_mask]
        #     con_emb_mask=concept_mask==self.concept_padding
        #
        #     con_user_emb=graph_con_emb
        #     con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.to(self.device))
        #
        #     '''这里的参数后续可以尝试与算ent时的参数合并'''
        #     user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb],dim=-1))
        #     uc_gate = F.sigmoid(self.gate_norm(user_emb))
        #     user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb
        #     entity_scores = F.linear(user_emb, db_nodes_features[:-1], self.output_en.bias)
        #     rec_loss_ = self.criterion2(entity_scores.unsqueeze(1).repeat(1, 32, 1).view(-1, entity_scores.shape[1]), ent_rec.view(-1))
        #     rec_loss = torch.sum(rec_loss_ * (ent_rec.view(-1) != self.entity_max))
        #
        # elif False:
        #     entity_emb = db_nodes_features[entity]
        #     db_m_emb = self.info_db_norm(entity_emb)
        #
        #
        #     concept_emb = con_nodes_features[concept_mask]
        #     con_m_emb=self.info_con_norm(concept_emb)
        #
        #
        #     attn_mask = -1e8 * (concept_mask == 0).unsqueeze(1).repeat(1,entity.shape[-1],1)
        #     attn_w = torch.matmul(db_m_emb,con_m_emb.transpose(1,2)) + attn_mask
        #     attn_w = torch.softmax(attn_w,dim=-1)
        #     entity_emb = entity_emb + torch.matmul(attn_w,concept_emb)
        #
        #     #todo path->gnode
        #     '''
        #     #when there exist multiple paths
        #     paths_ = paths.view(-1,paths.shape[-1])
        #     pmask= paths_ != self.entity_max
        #     paths_ = paths_ * pmask
        #     entity_emb_ = entity_emb.unsqueeze(1).repeat(1,paths.shape[1],1,1).view(-1,entity.shape[-1],self.dim)
        #     path_emb = entity_emb_.gather(1,paths_.unsqueeze(-1).repeat(1,1,self.dim))
        #     path_emb *= pmask.unsqueeze(-1)
        #     path_emb = path_emb.view(bsz,paths.shape[1],paths.shape[2],self.dim)
        #     path_emb = (path_emb * (paths != self.entity_max).unsqueeze(-1)).sum(dim=-2)
        #     path_mask = paths.sum(-1) != self.entity_max*paths.shape[-1]
        #
        #     state = self.state_norm(path_emb.view(bsz,-1))
        #     action_value_ = torch.matmul(path_emb,state.unsqueeze(1).transpose(1,2)).squeeze(-1)
        #     action_value_ = torch.softmax(action_value_ + (1-path_mask.int())*(-1e4),dim=-1)
        #     # cat = RelaxedOneHotCategorical(0.1, logits=action_value_)
        #     # action_value = cat.rsample().float()
        #
        #
        #     user_emb = (action_value_.unsqueeze(-1) * path_emb).sum(-2)
        #     entity_scores = F.linear(user_emb, db_nodes_features[:-1], self.output_en.bias)
        #     rec_loss_ = self.criterion2(entity_scores.unsqueeze(1).repeat(1, 32, 1).view(-1, entity_scores.shape[1]),ent_rec.view(-1))
        #     rec_loss = torch.sum(rec_loss_ * (ent_rec.view(-1) != self.entity_max))
        #     '''
        #     # paths_ = paths.view(-1,paths.shape[-1])
        #     # path_emb,_ = self.self_attn_batch_db(path_emb,paths_==self.entity_max)
        #     # path_emb = path_emb.view(bsz,-1,self.dim)
        #     paths_ = paths.view(-1, paths.shape[-1])
        #     pmask = paths_ != self.entity_max
        #     paths_ = paths_ * pmask
        #     entity_emb_ = entity_emb.unsqueeze(1).repeat(1, paths.shape[1], 1, 1).view(-1, entity.shape[-1], self.dim)
        #     path_emb = entity_emb_.gather(1, paths_.unsqueeze(-1).repeat(1, 1, self.dim))
        #     path_emb *= pmask.unsqueeze(-1)
        #     path_emb = path_emb.view(bsz, paths.shape[1], paths.shape[2], self.dim)
        #     path_emb = (path_emb * (paths != self.entity_max).unsqueeze(-1)).sum(dim=-2)
        #
        #     label_emb=db_nodes_features[ent_rec]
        #     # state = self.state_norm1(path_emb.view(bsz, -1))
        #     # action_value_ = torch.matmul(path_emb, state.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        #     # path_mask = paths.sum(-1) != self.entity_max * paths.shape[-1]
        #     # prior_rlv_score = torch.softmax(action_value_ + (1 - path_mask.int()) * (-1e4), dim=-1).unsqueeze(-1)
        #     # _,prior_rlv_score=self.self_attn_batch_db(path_emb,paths.sum(-1)==self.entity_max*paths.shape[-1])
        #     post_rlv_score_=torch.matmul(label_emb,path_emb.transpose(1,2))
        #     label_l = torch.sqrt((label_emb*label_emb).sum(-1)+1e-8).unsqueeze(-1)
        #     path_l = torch.sqrt((path_emb*path_emb).sum(-1)+1e-8).unsqueeze(-1)
        #
        #     post_div = torch.matmul(label_l,path_l.transpose(1,2))
        #     post_rlv_score = post_rlv_score_ / post_div
        #
        #     # path_mask = paths.sum(-1) == self.entity_max * paths.shape[-1]
        #     # post_path_mask = path_mask.unsqueeze(1).repeat(1,ent_rec.shape[-1],1)
        #     # post_rlv_score = post_path_mask*-1e8 + post_rlv_score
        #     # post_rlv_score = torch.softmax(post_rlv_score,-1)
        #
        #     if test==False:
        #         path_emb = self.dim2dimSe(path_emb)+path_emb
        #         path_emb = self.dim2dim(path_emb)
        #         user_emb_train = (post_rlv_score.unsqueeze(-1)*path_emb.unsqueeze(1)).sum(-2)
        #         entity_scores = F.linear(user_emb_train, db_nodes_features[:-1], self.output_en.bias)
        #
        #         # score_loss = self.score_loss(post_rlv_score,prior_rlv_score.squeeze(-1).unsqueeze(1).repeat(1,ent_rec.shape[-1],1)).sum()
        #         score_loss=0
        #         rec_loss_ = self.criterion2(entity_scores.view(-1,entity_scores.shape[-1]), ent_rec.view(-1))
        #         rec_loss = torch.sum(rec_loss_ * (ent_rec.view(-1) != self.entity_max))
        #     else:
        #         path_emb = self.dim2dimSe(path_emb) + path_emb
        #         path_emb = self.dim2dim(path_emb)
        #         user_emb_test = path_emb.sum(-2)
        #         entity_scores = F.linear(user_emb_test, db_nodes_features[:-1], self.output_en.bias)
        #         score_loss,rec_loss=0,0
        #
        # else:
        #
        #     concept_emb = con_nodes_features[concept_mask]
        #     concept_emb,_ = self.self_attn1(concept_emb,concept_mask==0)
        #
        #     cur_concept_emb = con_nodes_features[cur_concept_mask]
        #     cur_concept_emb, _ = self.self_attn1(cur_concept_emb, cur_concept_mask == 0)
        #
        #     paths_emb = db_nodes_features[paths].view(-1,paths.shape[2],self.dim)
        #     paths_mask1 = paths.view(-1,paths.shape[2]) == self.entity_max
        #     # cur_emb,_ = self.self_attn_batch_cur(cur_emb,cur_l_==self.entity_max)
        #     paths_emb,_ = self.self_attn_batch_his1(paths_emb,paths_mask1)
        #
        #     # m_paths_emb = db_nodes_features[m_paths].view(-1, m_paths.shape[2], self.dim)
        #     # m_paths_mask1 = m_paths.view(-1, m_paths.shape[2]) == self.entity_max
        #     # m_paths_emb, _ = self.self_attn_batch_his1(m_paths_emb, m_paths_mask1)
        #
        #     paths_emb_ = paths_emb.view(-1,self.dim)
        #     # _, _, score_loss = self.compute_barrow_loss(paths_emb_, paths_emb_, 0.1)
        #     # m_paths_emb_ = m_paths_emb.view(-1, self.dim)
        #     # _, _, score_loss2 = self.compute_barrow_loss(m_paths_emb_, m_paths_emb_, 0.1)
        #     # score_loss = ( score_loss2 + score_loss1 ) / 2
        #     if False:
        #         paths_emb = paths_emb.view(bsz,-1, self.dim)
        #         paths_mask2 = (paths.sum(-1) == self.entity_max*paths.shape[2]).float()
        #         path_emb,_ = self.self_attn_batch_his2(paths_emb,paths_mask2)
        #
        #         # m_paths_emb = m_paths_emb.view(bsz, -1, self.dim)
        #         # m_paths_mask2 = (m_paths.sum(-1) == self.entity_max * m_paths.shape[2]).float()
        #         # m_path_emb, _ = self.self_attn_batch_his2(m_paths_emb, m_paths_mask2)
        #     cur_emb = db_nodes_features[cur_l]
        #     cur_emb,_ = self.self_attn_batch_db(cur_emb,cur_l==self.entity_max)
        #     paths_emb = self.gate_layer1(paths_emb,cur_emb.unsqueeze(1).repeat(1,paths.shape[1],1).view(-1,self.dim))
        #     paths_emb = paths_emb.view(bsz, -1, self.dim)
        #     paths_mask2 = (paths.sum(-1) == self.entity_max * paths.shape[2]).float()
        #     user_his_emb, _ = self.self_attn_batch_his2(paths_emb, paths_mask2)
        #
        #     _, _, score_loss = self.compute_barrow_loss(paths_emb_, paths_emb_, 0.1)
        #     # _, _, info_db_loss1 = self.compute_barrow_loss(path_emb, concept_emb, 0.8)
        #     _, _, info_db_loss = self.compute_barrow_loss(cur_emb, cur_concept_emb, 0.8)
        #     # info_db_loss = info_db_loss1 + info_db_loss2
        #
        #     # path_emb_agg = self.gate_layer2(path_emb,m_path_emb)
        #     # user_his_emb = self.gate_layer1(concept_emb,path_emb)
        #     # user_his_emb = self.gate_layer2(concept_emb,path_emb,m_path_emb)
        #     # user_his_emb = self.gate_layer2(user_his_emb,sem_emb)
        #     #leaf
        #     # leaf_emb = db_nodes_features[leaf]
        #     # leaf_emb,_ = self.self_attn_batch_leaf(leaf_emb,leaf==self.entity_max)
        #     # user_his_emb = self.gate_layer_minus(user_his_emb,leaf_emb)
        #     user_cur_emb = self.gate_layer1(cur_concept_emb,cur_emb)
        #     # user_cur_emb = self.gate_layer2(user_cur_emb, cur_sem_emb)
        #
        #     user_emb = self.gate_layer3(user_his_emb, user_cur_emb)
        #
        #     entity_scores = F.linear(user_emb, db_nodes_features[:-1], self.output_en.bias)
        #
        #     rec_loss_ = self.criterion2(entity_scores.unsqueeze(1).repeat(1, 32, 1).view(-1, entity_scores.shape[1]),
        #                                 ent_rec.view(-1))
        #     rec_loss = torch.sum(rec_loss_ * (ent_rec.view(-1) != self.entity_max))
        #
        #     # _,_,info_db_loss = self.compute_barrow_loss(concept_emb,entity_emb,0.1)
        #
        # #
        # #
        # # #entity_scores = scores_db * gate + scores_con * (1 - gate)
        # # #entity_scores=(scores_db+scores_con)/2
        # #
        # # #mask loss
        # # #m_emb=db_nodes_features[labels.cuda()]
        # # #mask_mask=concept_mask!=self.concept_padding
        # # mask_loss=0#self.mask_predict_loss(m_emb, attention, xs, mask_mask.cuda(),rec.float())
        # # # related_con_emb=con_nodes_features[related_concept]
        # # # related_con_mask=related_concept==self.concept_padding
        # # # related_con_emb,_=self.self_attn(related_con_emb,related_con_mask)
        # # info_db_loss, info_con_loss=self.infomax_loss(con_nodes_features,db_nodes_features[:-1],con_user_emb,db_user_emb,con_label,db_label,db_con_mask)
        # #
        # #
        # # entity_scores=entity_scores.squeeze(1).squeeze(1)
        # # # rec_loss_=self.criterion(entity_scores.float(),movie_label)
        # # # rec_loss=torch.sum(rec_loss_*rec.float())
        # #
        # # rec_loss_ = self.criterion2(entity_scores.unsqueeze(1).repeat(1, 32, 1).view(-1, entity_scores.shape[1]), ent_rec.view(-1))
        # # rec_loss = torch.sum(rec_loss_ * (ent_rec.view(-1) != self.entity_max))
        #
        #
        #
        # # if test == False:
        # #     # con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        # #     # con_emb4gen = con_nodes_features4gen[concept_label_mask]
        # #     # con_mask4gen = concept_label_mask != self.concept_padding
        # #     # #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        # #     # kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.to(self.device))
        # #     kg_encoding=(None,None)
        # #     db_emb4gen=db_nodes_features[ent_rec] #batch*50*dim
        # #     db_mask4gen=ent_rec!=0
        # #     #db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        # #     db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.to(self.device))
        # #     # use teacher forcing
        # #     scores, preds,latent = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, mask_ys)
        # #     gen_loss = torch.mean(self.compute_loss(scores, mask_ys))
        # #
        # #     movie_rep = db_nodes_features[movie_label]
        # #     ent_rep=torch.cat([db_user_emb,movie_rep],dim=-1)
        # #     ent_token_rep=torch.gather(latent,dim=1,index=(ent_pos).unsqueeze(-1).repeat(1,1,self.embedding_size))
        # #     ent_token_rep=self.emb2ent(ent_token_rep)
        # #     ent_emb=torch.cat([ent_rep.repeat(1,ent_pos.shape[1],1).view(-1,self.dim*2),\
        # #                        ent_token_rep.view(-1,self.embedding_size)],dim=-1)
        # #     ent_res=self.ent2rec(ent_emb)
        # #     ent_scores=torch.matmul(ent_res,db_nodes_features.T)
        # #     ent_loss=self.criterion(ent_scores,ent_label.view(-1)).sum()
        # # else:
        # #     movie_pred_label=entity_scores[:,:6924].argmax(dim=-1)
        # #     movie_rep = db_nodes_features[movie_pred_label]
        # #     ent_rep = torch.cat([db_user_emb, movie_rep], dim=-1)
        # #     # ent_token_rep = torch.gather(latent, dim=1,
        # #     #                              index=(ent_pos).unsqueeze(-1).repeat(1, 1, self.embedding_size))
        # #     # ent_token_rep = self.emb2ent(ent_token_rep)
        # #     # ent_rep=ent_rep.repeat(1, ent_pos.shape[1], 1).view(-1, self.dim)
        # #     # ent_emb = torch.cat([ent_rep, torch.zeros(ent_rep.shape[0],self.embedding_size,device=self.device)], dim=-1)
        # #     # ent_res = self.ent2rec(ent_emb)
        # #     # ent_scores = torch.matmul(ent_res, db_nodes_features.T)
        # #
        # #
        # #     # con_nodes_features4gen = con_nodes_features  # self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        # #     # con_pred_label = torch.matmul(movie_rep,entity_scores)
        # #     # con_emb4gen = con_nodes_features4gen[ ]
        # #     # con_mask4gen = concept_label_mask != self.concept_padding
        # #     # kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        # #     # kg_encoding = (self.kg_norm(con_emb4gen), con_mask4gen.to(self.device))
        # #     kg_encoding=(None,None)
        # #
        # #     db_pred_label=entity_scores.sort(descending=True).indices[:,:4]
        # #     db_emb4gen = db_nodes_features[db_pred_label]  # batch*50*dim
        # #     db_mask4gen = db_pred_label != 0
        # #     # db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        # #     db_encoding = (self.db_norm(db_emb4gen), db_mask4gen.to(self.device))
        # #
        # #     scores, preds,incr_state = self.decode_greedy(
        # #         encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
        # #         bsz,
        # #         maxlen or self.longest_label
        # #     )
        # scores,preds,gen_loss,mask_loss=0,0,None,None
        # info_con_loss = 0
        # # score_loss =0
        # # info_db_loss =0
        '''这里可以添加往回填entity时重新用ent_token预测'''
        '''这里映射其他类型的entity'''
        # db_emb4gen=torch.stack(db_list4gen)
        # con_emb4gen,_=self.self_attn4gen(con_nodes_features[concept_mask],con_emb_mask.to(self.device))
        # user_emb4gen=self.user_norm4gen(torch.cat([db_emb4gen,con_emb4gen,user_emb],dim=-1))
        # # user_emb4gen=self.addgrammar(torch.cat([user_emb4gen_ent,encoder_states[0][:,special_pos]]))
        # entity_scores4gen=F.linear(user_emb4gen, db_nodes_features[:-1], self.output_en4gen.bias)
        #
        # entity_label4gen_=ent_rec.view(-1)
        # entity_label4gen=torch.where(entity_label4gen_<6924,self.entity_max,entity_label4gen_)
        # rec_loss4gen_=self.criterion2(entity_scores4gen.unsqueeze(1).repeat(1,32,1).view(-1,entity_scores4gen.shape[1]),entity_label4gen)
        # rec_loss4gen=torch.sum(rec_loss4gen_*(entity_label4gen!=self.entity_max))

        # for i in self.n_kind:
        #     total_loss.append(self.criterion(entity_scores,(labels==i).nonzero().view(-1)))
        # total_loss=sum(total_loss)
        '''这里建议再次加上预训练的loss'''
        '''从这里注释掉了'''
        # con_emb4gen=[]
        #
        # for i in range(self.n_kind):
        #     if test == False:
        #         # db_rd = (labels == i).float().topk(8).indices
        #         # print(db_rd.shape,(labels==i)[db_rd].shape)
        #         # mask4gen=torch.where((labels==i)[db_rd]==1,db_rd,self.entity_max)
        #         # db_mask4gen = mask4gen!=self.entity_max
        #         db_rd_=entity_label_vector[:,i,:]    #batch,20
        #         db_mask4gen_=db_rd_==self.entity_max
        #
        #     else:
        #         db_rd_ = (entity_scores * (self.entity_label == i+1)).topk(20).indices
        #         db_mask4gen_ = db_rd_ == self.entity_max
        #     # if test==False:
        #     #     db_emb4gen_batch_ = db_nodes_features[entity_label_vector[i]]
        #     #     db_rd_=entity_label_vector[i]==self.entity_max
        #     # else
        #     db_emb4gen_batch_=db_nodes_features[db_rd_]
        #     db_emb4gen,_=self.self_attn_batch_db(db_emb4gen_batch_,db_mask4gen_)
        #     con_emb4gen.append(db_emb4gen)
        # con_emb4gen=torch.stack(con_emb4gen).transpose(0,1) # bsz,n_kind,emb_size
        #
        # con_label=F.sigmoid(torch.matmul(con_emb4gen,con_nodes_features.transpose(-1,-2))) #bsz,n_kind,con_nodes-num
        #
        # con_score=torch.zeros(xs.shape[0],con_label.shape[-1],device=self.device)
        # for i in range(xs.shape[0]):
        #     for j,key in enumerate(self.n_kind_set.keys()):
        #         for rd in self.n_kind_set[key]:
        #             if rd in xs[i]:
        #                 con_score[i]+=con_label[i][j]
        #                 break
        #
        # if True:
        #     word_logit_=torch.zeros(con_emb4gen.shape[0],self.n_kind,self.embeddings.weight.shape[0],device=self.device)
        #     word_logit_[:,:,torch.tensor(self.con2word,device=self.device)]=con_label
        #     word_logit_[:,:,0]=0
        #     for k,v in self.n_kind_set.items():
        #         word_logit_[:,:,v]=0
        #     word_logit=word_logit_.topk(50).indices.view(xs.shape[0],-1)#bsz*n_kind,50
        #     # word_score=F.softmax(word_logit_.topk(50).values.view(xs.shape[0],-1),dim=-1)
        #
        #     # word_emb=self.embeddings(word_logit)
        #     #
        #     # kind_emb=self.kind_emb.weight.unsqueeze(1).repeat(xs.shape[0],50,1)
        #     # word_emb=self.wordfusekind(torch.cat([word_emb,kind_emb],dim=-1)).view(xs.shape[0],self.n_kind*50,-1)
        #
        #
        #     #fusion based
        #     # word_emb=self.semantic_drag(self.self_attn_word((self.embeddings(word_logit)),word_logit==0)[0].view(xs.shape[0],self.n_kind,self.embedding_size))
        #     word_emb=self.semantic_drag(self.encoder.embeddings(word_logit))
        #     db_encoding=(word_emb,word_logit!=0)
        # else:
        #     con_rd=con_score.topk(50).indices.view(xs.shape[0],-1)#bsz,n_kind,100
        #     db_encoding=(self.kg_norm(con_nodes_features[con_rd]),con_rd!=self.concept_padding)
        #     con_emb=self.self_conattn(con_nodes_features[con_rd.view(-1,50)],con_rd.view(-1,50)==0)[0]#bsz*n_kind,dim
        #     word_embedding=self.semantic_drag_new(self.embeddings.weight)
        #     word_logit=torch.matmul(con_emb,word_embedding.T)#bsz*n_kind,word_num
        #     word_rd=word_logit.topk(50).indices
        #     word_emb=word_embedding[word_rd]
        #     # db_encoding=(self.kg_norm(word_emb).view(xs.shape[0],-1,self.embedding_size),word_rd.view(xs.shape[0],-1)!=0)
        #     # word_emb_=self.self_attn_word_new(word_embedding[word_rd],word_rd==0)[0].view(xs.shape[0],self.n_kind,-1)
        #     # word_emb=self.kg_norm(word_emb_)
        # # word_logit=F.sigmoid(word_logit_) #bsz,n_kind,word_num     word_num*emb_size
        # # # word_emb = (word_logit.unsqueeze(-1) * self.embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(con_label.shape[0], self.n_kind,1, 1)).sum(-2)/word_logit.sum(-1).unsqueeze(-1)
        # # word_emb=torch.matmul(word_logit,self.embeddings.weight)/word_logit.sum(-1).unsqueeze(-1)
        #
        # # # bsz,n_kind,emb_size
        # # kind_logit_ = self.wordfusekind(torch.cat([word_emb, self.kind_emb.weight.unsqueeze(0).repeat(xs.shape[0], 1, 1)], dim=-1))  # bsz,n_kind,emb
        # # kind_emb = kind_logit_ + word_emb + self.kind_emb.weight.unsqueeze(0).repeat(xs.shape[0], 1, 1)
        #
        # # con_nodes_features4gen = con_nodes_features  # self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        # # con_emb4gen = con_nodes_features4gen[concept_mask]
        # # con_mask4gen = concept_mask != self.concept_padding
        # # # kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        # # kg_encoding = (self.kg_norm(con_emb4gen), con_mask4gen.to(self.device))
        # #
        # # db_emb4gen = db_nodes_features[entity_label_vector.long().view(xs.shape[0],-1)]  # batch*50*dim
        # # db_mask4gen = entity_label_vector.view(xs.shape[0],-1) != 0
        # # # db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        # # db_encoding = (self.db_norm(db_emb4gen), db_mask4gen.to(self.device))
        #
        # if test==False:
        #     scores, preds,latent = self.decode_forced_new(encoder_states, mask_ys, word_emb,db_user_emb,con_user_emb,db_encoding)
        #     # # latent=latent[:,1:]
        #     # # special_mask=special_mask[:,1:]
        #     # if self.classify==True:
        #     #     special_mask_sparse=torch.where(special_mask==6,0,special_mask).to_sparse().coalesce()
        #     #     # print(latent[special_mask_sparse.indices()[0]].shape,special_mask_sparse.indices()[1].shape)
        #     #     class_latent=torch.gather(latent[special_mask_sparse.indices()[0]],dim=1,index=special_mask_sparse.indices()[1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.embedding_size)).squeeze(1)
        #     #     class_label=special_mask_sparse.values()-1
        #     #     special_sum=(special_mask!=0).sum(-1)
        #     #     infor_total=[]
        #     #     divisor = encoder_states[1].type_as(encoder_states[0]).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
        #     #     output = encoder_states[0].sum(dim=1) / divisor
        #     #     for i,sp_num in enumerate(special_sum):
        #     #         infor_total.extend([output[i]]*sp_num)
        #     #     try:
        #     #         infor_total=torch.stack(infor_total)
        #     #     except:
        #     #         assert special_sum.sum()==0
        #     #     try:
        #     #         assert len(infor_total)==len(class_label)
        #     #     except:
        #     #         print(infor_total.shape,class_label.shape)
        #     #     if special_sum.sum()!=0:
        #     #         class_info=torch.cat([class_latent,infor_total],dim=-1)
        #     #         class_pred=self.classlinear1(class_info)
        #     #         class_pred[:,2]=0
        #     #         class_loss=self.criterion3(class_pred,class_label).sum()
        #     #     else:
        #     #         class_loss=rec_loss*0
        #     # else:
        #     #     class_loss=None
        #     gen_loss = torch.mean(self.compute_loss(scores, mask_ys))
        #     # class_score=None
        # else:
        #     # special_max_scores=[entity_scores[:,30434:30452].max(-1).values,
        #     #                     entity_scores[:,30459:30471].max(-1).values,
        #     #                     torch.zeros(entity_scores.shape[0],device=self.device),
        #     #                     entity_scores[:,19727:30434].max(-1).values,
        #     #                     entity_scores[:,6924:19727].max(-1).values]
        #     #                     # torch.zeros(entity_scores.shape[0],device=self.device)] #nkind,bsz
        #     # special_max_scores=torch.stack(special_max_scores)
        #     # # divisor = encoder_states[1].type_as(encoder_states[0]).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
        #     # # output = encoder_states[0].sum(dim=1) / divisor
        #     scores, preds, = self.decode_greedy(
        #         encoder_states, word_emb,
        #         xs.shape[0],
        #         maxlen or self.longest_label,
        #         db_user_emb,con_user_emb,db_encoding
        #     )
        #
        #     gen_loss = None
        #     # class_loss=None

        # #entity_scores = F.softmax(entity_scores.cuda(), dim=-1).cuda()
        #
        # rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        # #rec_loss=self.klloss(entity_scores.squeeze(1).squeeze(1).float(), labels.float().cuda())
        # rec_loss = torch.sum(rec_loss*rec.float().cuda())
        #
        # self.user_rep=user_emb


        #generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen=con_nodes_features
        con_emb4gen = con_nodes_features4gen[concept]
        con_mask4gen = concept != 0
        kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda())
        con_user_emb,_ = self.self_conattn(con_emb4gen,concept==0)

        db_emb4gen=db_nodes_features[entity]
        db_mask4gen= entity != self.entity_max
        db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda())
        db_user_emb,_ = self.self_attn_word_new(db_emb4gen,entity==self.entity_max)

        if test == False:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, ys)
            gen_loss = torch.mean(self.compute_loss(scores, ys))

        else:
            scores, preds = self.decode_greedy(
                encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,bsz,
                maxlen or self.longest_label
            )
            gen_loss = None
        entity_scores, rec_loss, score_loss, mask_loss, info_db_loss, info_con_loss = 0,0,0,0,0,0
        return scores, preds, entity_scores, rec_loss, score_loss, gen_loss, mask_loss, info_db_loss, info_con_loss

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]
x
        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.to(self.device), score_view.to(self.device))
        return loss

    def save_model(self,name=''):
        if name!='':
            torch.save(self.state_dict(), 'saved_model/'+name+'.pkl')
        else:
            torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')

    def load_model(self,name=''):
        params=[self.embeddings.parameters()]
        for param in params:
            for pa in param:
                pa.requires_grad = False
        if name=='':
            self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))
        else:
            self.load_state_dict(torch.load('saved_model/'+name+'.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsque
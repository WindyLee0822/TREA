#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import signal
import json
import argparse
import pickle as pkl
from dataset import CRSdataset,gen_CRSdataset,dataset4tgredial,CRSdataset4tgredial
from model import CrossModel
from generate_model import gen_CrossModel
import torch.nn as nn
import random
from torch import optim
import torch
import wandb
# try:
import torch.version
import torch.distributed as dist
TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args4tgredial():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=64)
    train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=str,default='rec')
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=30)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.022)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)
    train.add_argument("-movie_max","--movie_max",type=int,default=6924)

    train.add_argument("-n_heads","--n_heads",type=int,default=2)
    train.add_argument("-n_layers","--n_layers",type=int,default=2)
    train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    train.add_argument("-n_entity","--n_entity",type=int,default=54790)
    # train.add_argument("-n_relation","--n_relation",type=int,default=214)
    train.add_argument("-n_concept","--n_concept",type=int,default=65580)
    train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-dim","--dim",type=int,default=128)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train
# co_edge = json.load(open('co_edge.json'))
# relations_names = ['time', 'director', 'starring', 'genre', 'subject', 'belong', 'timeR', 'directorR', 'starringR','genreR', 'subjectR', 'belongR']
# device='cuda'
# def add_edge(delete_nodes):
#     edge_index = [[], []]
#     edge_type = []
#     for e in co_edge:
#         if e[0] in delete_nodes or e[1] in delete_nodes:
#             continue
#         edge_index[0].append(e[0])
#         edge_index[1].append(e[1])
#         edge_type.append(len(relations_names))
#     edge_index = torch.from_numpy(np.array(edge_index)).to(device)
#     edge_type = torch.from_numpy(np.array(edge_type)).long().to(device)
#     return edge_index,edge_type

def seed_initial(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False

class TrainLoop_fusion_gen():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.train_dataset=dataset('data',opt,mode='gen_train')
        self.device='cuda'
        self.entity_max=opt['n_entity']
        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}
        self.con2word = self.train_dataset.con2word
        self.n_kind_set = self.train_dataset.n_kind_set
        self.special_set={}
        for k,v in self.n_kind_set.items():
            for i in v:
                self.special_set[i]=k
        # self.ENT_rd=self.dict['__ENT__']
        # self.MOVIE_rd=self.dict['__MOVIE__']
        self.ent2mov={v:k for k,v in self.train_dataset.movieid2entityid.items()}
        self.nodes=json.load(open('data/redial_kg.json'))['nodes']


        self.movie_max=self.train_dataset.entity_max
        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = [i for i in range(6924)]

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune=True)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = gen_CrossModel(self.opt, self.dict, self.con2word, self.n_kind_set, is_finetune).to(self.device)
        if self.opt['embedding_type'] != 'random':
            pass
        # if self.use_cuda:
        #     self.model.cuda()

    def train(self):
        self.model.load_model('best_bleu')
        # rec_dict=torch.load()
        # model_dict = self.model.state_dict()
        # ll=['representation_bias.bias',
        #     'user_representation_to_bias_2.weight','user_representation_to_bias_2.bias','self_attn_word.a',
        #     'self_attn_word.b',
        #     'encoder.embeddings.weight','decoder.embeddings.weight','embeddings.weight',
        #     'info_output_word.bias','info_output_word.weight','representation_bias.weight']
        # # 筛除不加载的层结构
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ll}
        # # pretrained_dict['encoder.embeddings']=pretrained_dict['embeddings']
        # # pretrained_dict['decoder.embeddings'] = pretrained_dict['embeddings']
        # # 更新当前网络的结构字典
        # model_dict.update(pretrained_dict)
        # self.model.load_state_dict(rec_dict,strict=False)

        losses=[]
        best_val_dist={'dist4':0}
        best_val_bleu={'bleu3': 0}

        gen_stop=False
        for i in range(self.epoch*44):
            train_set=gen_CRSdataset(self.train_dataset.gen_data_process(True),self.opt['n_entity'],self.opt['n_concept'],self.dict)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context, response, rel_ent, concept in tqdm(train_dataset_loader):
                batch_size = context.shape[0]
                # seed_sets = []
                # for b in range(batch_size):
                #     seed_set = entity[b].nonzero().view(-1).tolist()
                #     seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss,score_loss, gen_loss, mask_loss, info_db_loss, _  = self.model(
                    context.to(self.device), response.to(self.device), rel_ent.to(self.device), concept.to(self.device),test=False)


                joint_loss=gen_loss


                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%150==0:
                    print('gen loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    # print('class loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                    losses=[]
                num+=1
            flag=0
            output_metrics_gen,infer_text = self.val(True)
            if best_val_dist["dist4"] >= output_metrics_gen["dist4"]:
                pass
            else:
                flag=1
                best_val_dist = output_metrics_gen
                self.model.save_model('best_genwo')
                self.write_file('gen',infer_text)
                print("generator model saved once best_gen------------------------------------------------")
            if best_val_bleu["bleu3"] >= output_metrics_gen["bleu3"]:
                pass
            else:
                flag=1
                best_val_bleu = output_metrics_gen
                self.model.save_model('best_bleuwo')
                self.write_file('bleu', infer_text)
                print("generator model saved once best_bleu------------------------------------------------")
            # if best_val_ent["ent_recall@50"] >= output_metrics_gen["ent_recall@50"]:
            #     pass
            # else:
            #     flag=1
            #     best_val_ent = output_metrics_gen
            #     self.model.save_model('best_ent')
            #     self.write_file('ent', infer_text)
            #     print("generator model saved once best_ent------------------------------------------------")
            # if flag==0:
            #     self.write_file('tmp',infer_text)
        _=self.val(is_test=True)
        print(best_val_dist)
        print(best_val_bleu)

    def val(self,is_test=False):
        # self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0,
        #                   "recall@1":0,"recall@10":0,"recall@50":0,"ent_recall@1": 0, "ent_recall@10": 0, "ent_recall@50": 0,'*r_count':0,'r_count':0,'ent_count':0,
        #                   "*recall@1":0,"*recall@10":0,"*recall@50":0,'ent_mention_times':0,'movie_mention_times':0}
        self.metrics_gen = {"ppl": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "bleu1": 0, "bleu2": 0,
                            "bleu3": 0, "bleu4": 0, "count": 0, 'movie_mention_times': 0}

        self.model.eval()
        val_dataset = dataset('data', self.opt,mode='gen_test')
        val_set=gen_CRSdataset(val_dataset.gen_data_process(True),self.opt['n_entity'],self.opt['n_concept'],self.dict)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        inference_sum=[]
        infer_text=[]
        golden_sum=[]
        losses=[]
        for context, response, rel_ent, concept in tqdm(val_dataset_loader):
            with torch.no_grad():
                batch_size = context.shape[0]
                _, _, _, _,_, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(
                    context.to(self.device), response.to(self.device), rel_ent.to(self.device), concept.to(self.device),test=False)
                scores, preds, rec_scores, rec_loss,score_loss, _, mask_loss, info_db_loss, info_con_loss =self.model(
                    context.to(self.device), response.to(self.device), rel_ent.to(self.device), concept.to(self.device),test=True)

            golden_sum.extend(self.vector2sentence(response.cpu(),text=False))
            inference_sum.extend(self.vector2sentence(preds.cpu(),text=False))
            # infer_text.extend(self.vector2sentence(preds.cpu(),rec_scores.cpu()))
            # context_sum.extend(self.vector2sentence(context.cpu(),text=False))
            # recs.extend(rec.cpu())
            # movies.extend(movie_rec.cpu())
            # rec_score.extend(rec_scores.cpu())
            losses.append(torch.mean(gen_loss))
            # class_scores.extend(class_score.cpu())
            # ent_label.extend(entity_label_vector.cpu())
            #print(losses)
            #exit()

        self.metrics_cal_gen(losses,inference_sum,golden_sum)
        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            elif 'ent_recall' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['ent_count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        print(output_dict_gen)
        return output_dict_gen,inference_sum
    def write_file(self,name,inference_sum):
        f=open('output_test_'+name+'.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        f.close()

    def metrics_cal_ent(self,scores,ent_labels):
        self.metrics_rec_ent = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "gate": 0, "count": 0,
                            'gate_count': 0}
        batch_size = scores.shape[0]
        outputs = scores.cpu()
        outputs[:, torch.LongTensor(self.movie_ids)]=0
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            for ent in ent_labels[b]:
                if ent == self.entity_max:
                    continue
                target_idx = ent
                self.metrics_rec_ent["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
                self.metrics_rec_ent["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
                self.metrics_rec_ent["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
                self.metrics_rec_ent["count"] += 1
    def metrics_cal_gen(self,rec_loss,preds,responses):

        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)#unigram_count
            dis2 = len(bigram_set) / len(outs)#bigram_count
            dis3 = len(trigram_set)/len(outs)#trigram_count
            dis4 = len(quagram_set)/len(outs)#quagram_count
            return dis1, dis2, dis3, dis4


        self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
        generated=[]

        for out, tar in zip(preds, responses):
            bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1
            #
            # if rec==1:
            #     _, pred_idx = torch.topk(score, k=100, dim=0)
            #     for target in movie:
            #         if target==self.movie_max:
            #             continue
            #         self.metrics_rec["recall@1"] += int(target in pred_idx[:1].tolist())
            #         self.metrics_rec["recall@10"] += int(target in pred_idx[:10].tolist())
            #         self.metrics_rec["recall@50"] += int(target in pred_idx[:50].tolist())
            #         self.metrics_rec["count"]+=1
        dis1, dis2, dis3, dis4=distinct_metrics(generated)
        self.metrics_gen['dist1']=dis1
        self.metrics_gen['dist2']=dis2
        self.metrics_gen['dist3']=dis3
        self.metrics_gen['dist4']=dis4

        # todo 这里的ent_label_vetcors需要转置成bsz,n_kind,num
        for out in zip(preds):
            for word in out:
                if '__MOVIE__' in word:
                    self.metrics_gen['movie_mention_times']+=1

        # for out,tar,rec_score,ent_label_vector in zip(preds,responses,rec_scores,ent_label_vectors):
        #     cur_movie=0
        #     for word in tar:
        #         if '__MOVIE__' in word:
        #             cur_movie+=1
        #     movie_num=len(list(set(ent_label_vector[5].tolist())))-1
        #     movie_cal_num=min(cur_movie,movie_num)
        #     self.metrics_gen['r_count'] += movie_cal_num
        #     self.metrics_gen['*r_count']+=cur_movie
        #     c=0
        #     for word in out:
        #         if '__MOVIE__' in word:
        #             c+=1
        #     self.metrics_gen['movie_mention_times']+=c
        #     if c!=0:
        #         mul=min(c,movie_cal_num)
        #         r=[]
        #         outputs = rec_score[torch.LongTensor(self.movie_ids)]
        #         _, pred_idx = torch.topk(outputs, k=100, dim=0)
        #         for movie in ent_label_vector[5]:
        #             if movie ==self.entity_max:
        #                 continue
        #             target_idx = self.movie_ids.index(movie)
        #             #     r.append((int(target_idx in pred_idx[:1].tolist()),int(target_idx in pred_idx[:10].tolist()),int(target_idx in pred_idx[:50].tolist())))
        #             # r_sum=[sum(i) for i in r]
        #             # for i in range(min(c,movie_cal_num)):
        #             #     cur_id=r_sum.index(max(r_sum))
        #             self.metrics_gen["recall@1"] += int(target_idx in pred_idx[:1].tolist())*mul
        #             self.metrics_gen["recall@10"] += int(target_idx in pred_idx[:10].tolist())*mul
        #             self.metrics_gen["recall@50"] += int(target_idx in pred_idx[:50].tolist())*mul
        #             self.metrics_gen["*recall@1"] += int(target_idx in pred_idx[:1].tolist()) * c
        #             self.metrics_gen["*recall@10"] += int(target_idx in pred_idx[:10].tolist()) * c
        #             self.metrics_gen["*recall@50"] += int(target_idx in pred_idx[:50].tolist()) * c
        #             # r_sum[cur_id]=0
        #     # total_ent=[]
        #     # for ii in [0,1,3,4]:
        #     #     total_ent.extend(ent_label_vector[ii].tolist())
        #     # total_ent=list(set(total_ent))
        #     #
        #     # # ent_num=len(list(set(ent_label_vector[4].tolist())))-1
        #     # ent_num=len(total_ent)-1
        #     # self.metrics_gen['ent_count']+=ent_num
        #     # c=out.count('__ENT__')
        #     # if c!=0:
        #     #     for ent_id in range(min(ent_num,c)):
        #     #         ent_pos=out.index('__ENT__')
        #     #         out[ent_pos]=''
        #     #         cur_class_score=class_score[ent_pos]
        #     #         ent_score_mul = [0] * 6924 + [cur_class_score[4]] * 12803 + [cur_class_score[3]] * 10707 + [cur_class_score[0]] * 18 + [0] * 7 + [cur_class_score[1]] * 12
        #     #         outputs = rec_score * ent_score_mul
        #     #         _, pred_idx = torch.topk(outputs, k=100, dim=0)
        #     #         for ent in total_ent:
        #     #             if ent == self.entity_max:
        #     #                 continue
        #     #             target_idx = ent
        #     #             # r.append((int(target_idx in pred_idx[:1].tolist()),int(target_idx in pred_idx[:10].tolist()),int(target_idx in pred_idx[:50].tolist())))
        #     #             self.metrics_gen["ent_recall@1"] += int(target_idx in pred_idx[:1].tolist())
        #     #             self.metrics_gen["ent_recall@10"] +=int(target_idx in pred_idx[:10].tolist())
        #     #             self.metrics_gen["ent_recall@50"] += int(target_idx in pred_idx[:50].tolist())
        #     ent_num = len(list(set(ent_label_vector[4].tolist()))) - 1
        #     self.metrics_gen['ent_count'] += ent_num
        #     c=0
        #     for word in out:
        #         if '__PERSON__' in word:
        #             c+= 1
        #     self.metrics_gen['ent_mention_times']+=c
        #     if c!=0:
        #         mul=min(c,ent_num)
        #         outputs = rec_score[6924:19727]
        #         _, pred_idx = torch.topk(outputs, k=100, dim=0)
        #         pred_idx=pred_idx+6924
        #         for ent in ent_label_vector[4]:
        #             if ent ==self.entity_max:
        #                 continue
        #             target_idx = ent
        #             self.metrics_gen["ent_recall@1"] += int(target_idx in pred_idx[:1].tolist())*mul
        #             self.metrics_gen["ent_recall@10"] += int(target_idx in pred_idx[:10].tolist())*mul
        #             self.metrics_gen["ent_recall@50"] += int(target_idx in pred_idx[:50].tolist())*mul
        #     ent_num=len(list(set(ent_label_vector[3].tolist()))) - 1
        #     self.metrics_gen['ent_count']+=ent_num
        #     c = 0
        #     for word in out:
        #         if '__SUBJECT__' in word:
        #             c+=1
        #     self.metrics_gen['ent_mention_times'] += c
        #     if c != 0:
        #         mul=min(c,ent_num)
        #         outputs = rec_score[19727:30434]
        #         _, pred_idx = torch.topk(outputs, k=100, dim=0)
        #         pred_idx = pred_idx + 19727
        #         for ent in ent_label_vector[3]:
        #             if ent == self.entity_max:
        #                 break
        #             target_idx = ent
        #             self.metrics_gen["ent_recall@1"] += int(target_idx in pred_idx[:1].tolist())*mul
        #             self.metrics_gen["ent_recall@10"] += int(target_idx in pred_idx[:10].tolist())*mul
        #             self.metrics_gen["ent_recall@50"] += int(target_idx in pred_idx[:50].tolist())*mul
        #     ent_num=len(list(set(ent_label_vector[0].tolist()))) - 1
        #     self.metrics_gen['ent_count'] += ent_num
        #     c = 0
        #     for word in out:
        #         if '__GENRE__' in word:
        #             c+=1
        #     self.metrics_gen['ent_mention_times'] += c
        #     if c != 0:
        #         mul=min(c,ent_num)
        #         # ent_score_mul = [0] * 6924 + [class_score[4]] * 12803 + [class_score[3]] * 10707 + [class_score[0]] * 18 + [0] * 7 + [class_score[1]] * 12
        #         outputs = rec_score[30434:30452]
        #         _, pred_idx = torch.topk(outputs, k=18, dim=0)
        #         pred_idx = pred_idx + 30434
        #         for ent in ent_label_vector[0]:
        #             if ent == self.entity_max:
        #                 break
        #             target_idx = ent
        #             self.metrics_gen["ent_recall@1"] += int(target_idx in pred_idx[:1].tolist())*mul
        #             self.metrics_gen["ent_recall@10"] += int(target_idx in pred_idx[:10].tolist())*mul
        #             self.metrics_gen["ent_recall@50"] += int(target_idx in pred_idx[:50].tolist())*mul
        #     ent_num=len(list(set(ent_label_vector[1].tolist()))) - 1
        #     self.metrics_gen['ent_count'] += ent_num
        #     c = 0
        #     for word in out:
        #         if '__TIME__' in word:
        #             c+=1
        #     self.metrics_gen['ent_mention_times'] += c
        #     if c != 0:
        #         mul=min(c,ent_num)
        #         outputs = rec_score[30459:30471]
        #         _, pred_idx = torch.topk(outputs, k=12, dim=0)
        #         pred_idx = pred_idx + 30459
        #         for ent in ent_label_vector[1]:
        #             if ent == self.entity_max:
        #                 break
        #             target_idx = ent
        #             self.metrics_gen["ent_recall@1"] += int(target_idx in pred_idx[:1].tolist())*mul
        #             self.metrics_gen["ent_recall@10"] += int(target_idx in pred_idx[:10].tolist())*mul
        #             self.metrics_gen["ent_recall@50"] += int(target_idx in pred_idx[:50].tolist())*mul
        #
        #
        #     '''
        #     0-6923 movie
        #     6924-19726 person
        #     19727-30433 subject
        #     30434-30451 genre
        #     30452-30458 attr
        #     30459-30470 time
        #     '''

    def vector2sentence(self,batch_sen,rec_scores=None,text=True):
        sentences=[]
        mention_rd=0

        if text==True:
            for sen,rec_score in zip(batch_sen.numpy().tolist(),rec_scores):
                # ent_num=1
                # movie_num=1
                sentence=[]
                special_rd = [1] * 7

                for word in sen:
                    special_rd[6]=1
                    flag=0
                    for k,v in self.n_kind_set.items():
                        if word in v:
                            flag=1
                            break

                    if flag:
                        if word in self.n_kind_set['__MOVIE__']:
                            ent_score = rec_score[:6924]
                            ent_topk=ent_score.topk(special_rd[6]).indices.numpy().tolist()
                            cur_node=ent_topk[special_rd[6]-1]
                            sentence.append('@'+str(self.ent2mov[cur_node]))
                            special_rd[6]+=1
                            # elif word ==self.ENT_rd:
                            #     '''
                            #                             0-6923 movie
                            #                             6924-19726 person
                            #                             19727-30433 subject
                            #                             30434-30451 genre
                            #                             30452-30458 attr
                            #                             30459-30470 time
                            #     '''
                            #     ent_score_mul=[0]*6924+[class_score[4]]*12803+[class_score[3]]*10707+[class_score[0]]*18+[0]*7+[class_score[1]]*12
                            #     ent_score=rec_score*ent_score_mul
                            #     ent_topk=ent_score.topk(1).indices.numpy().tolist()
                            #     cur_node=ent_topk[0]
                            #     sentence.extend(self.nodes[cur_node]['name'].replace('_', ' ').split())
                            #     # if self.special_set[word]=='__MOVIE__':
                            #     #     ent_score=rec_score[:6924]
                            #     #     ent_topk=ent_score.topk(special_rd[0]).indices.numpy().tolist()
                            #     #     cur_node=ent_topk[special_rd[0]-1]
                            #     #     sentence.append('@'+self.ent2mov[cur_node])
                        else:
                            label=self.special_set[word]
                            if label=='__PERSON__':
                                ent_score = rec_score[6924:19727]
                                ent_topk = ent_score.topk(special_rd[1]).indices.numpy().tolist()
                                cur_node = ent_topk[special_rd[1] - 1]
                                sentence.extend(self.nodes[cur_node]['name'].replace('_',' ').split())
                                special_rd[1]+=1
                            elif label=='__SUBJECT__':
                                ent_score = rec_score[19727:30434]
                                ent_topk = ent_score.topk(special_rd[2]).indices.numpy().tolist()
                                cur_node = ent_topk[special_rd[2] - 1]
                                sentence.extend(self.nodes[cur_node]['name'].replace('_', ' ').split())
                                special_rd[2]+=1
                            elif self.special_set[word]=='__GENRE__':
                                ent_score=rec_score[30434:30452]
                                ent_topk = ent_score.topk(min(special_rd[3],18)).indices.numpy().tolist()
                                cur_node = ent_topk[min(special_rd[3] - 1,17)]
                                sentence.extend(self.nodes[cur_node]['name'].lower().replace('_', ' ').split())
                                #todo 这里不应该加
                                special_rd[3]+=1
                            elif self.special_set[word] == '__TIME__':
                                ent_score = rec_score[30459:30470]
                                ent_topk = ent_score.topk(min(special_rd[5] ,11)).indices.numpy().tolist()
                                cur_node = ent_topk[min(special_rd[5] - 1,10)]
                                sentence.extend(self.nodes[cur_node]['name'].lower().replace('_', ' ').split())
                                special_rd[5]+=1
                    elif word>3:
                        sentence.append(self.index2word[word])
                    elif word==3:
                        sentence.append('_UNK_')
                sentences.append(sentence)
        else:
            for sen in batch_sen.numpy().tolist():
                sentence=[]
                for word in sen:
                    if word>3:
                        sentence.append(self.index2word[word])
                    elif word==3:
                        sentence.append('_UNK_')
                sentences.append(sentence)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

class TrainLoop_fusion_rec4tgredial():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.train_dataset=dataset4tgredial('data',opt,mode='train')
        self.device='cuda'
        self.entity_max=opt['n_entity']
        self.dict=self.train_dataset.word2index
        # self.con2word=self.train_dataset.con2word
        # self.n_kind_set=self.train_dataset.n_kind_set
        self.index2word={self.dict[key]:key for key in self.dict}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']
        self.movie_max=self.train_dataset.entity_max
        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = json.load(open('tgredial/movie_ids.json'))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)
        self.rec_result = []
        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict,0,0, is_finetune,is_tgredial=True).to(self.device)
        if self.opt['embedding_type'] != 'random':
            pass
        # if self.use_cuda:
        #     self.model.to(self.device)

    def train(self):
        # self.model.load_model()
        losses=[]
        best_val_rec=0
        rec_stop=False
        for i in range(0):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'],self.dict)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context, response, entity_vec, entity_vector, movie, \
                concept_mask, concept_vec, entity_rec in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, ent_loss, gen_loss, mask_loss, info_db_loss, _ = self.model(
                        context.to(self.device), response.to(self.device),
                        concept_mask, seed_sets, entity_vector.to(self.device), concept_vec, entity_vec.to(self.device),
                        entity_rec.to(self.device), test=False)
                joint_loss=info_db_loss#+info_con_loss

                losses.append([info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('info db loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    #print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

        # self.model.save_model(name='pre_modifyattn')
        # print("masked loss pre-trained")
        # losses=[]
        # self.model.load_model('net_parameter4tgredial')
        # self.val()
        # raise ValueError
        for i in range(self.epoch*8):
            self.rec_result = []
            train_set=CRSdataset4tgredial(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'],self.dict)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            # self.val()
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity_vec,entity_vector,movie,\
            #     concept_mask,concept_label_mask,dbpedia_mask,concept_vec, db_vec,rec,labels,entity_label_vector,rec_mask,entity_rec,\
            #     special_mask,related_concept,word_set,ent_pos,ent_label in tqdm(train_dataset_loader):
            for context, cur_context,response, entity_vec, entity_vector, movie, \
                concept_mask,cur_concept_mask, concept_vec, entity_rec,branch,m_branch,cur_l,leaf,i_rd in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)

                # add_edge_idx, add_edge_type = add_edge(entity_vector.view(-1).tolist())
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, score_loss, gen_loss, mask_loss, info_db_loss, _ = self.model(
                    context.to(self.device),cur_context.to(self.device), response.to(self.device),
                    concept_mask.to(self.device),cur_concept_mask.to(self.device), seed_sets, entity_vector.to(self.device), concept_vec.to(self.device), entity_vec.to(self.device),
                    entity_rec.to(self.device),branch.to(self.device),m_branch.to(self.device),cur_l.to(self.device), leaf.to(self.device),test=False)

                joint_loss = rec_loss+0.002*score_loss+0.008*info_db_loss

                losses.append([rec_loss,info_db_loss,score_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%150==0:
                    print('rec loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('info loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    print('score loss is %f' % (sum([l[2] for l in losses]) / len(losses)))
                    losses=[]
                num+=1

            output_metrics_rec = self.val()

            if best_val_rec > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]:
                rec_stop=True
            else:
                best_val_rec = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]
                # self.model.save_model()
                with open('rec_result.json','w') as f:
                    json.dump(self.rec_result,f)
                print("recommendation model saved once------------------------------------------------")

        _=self.val(is_test=True)

    def metrics_cal_rec(self,rec_loss,scores,labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()

        # for b in range(batch_size):
        #     l=[i for i in mentioned[b] if i<6924]
        #     outputs[b,l] = -99999
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            if labels[b].item()==self.movie_max:
                raise ValueError
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.rec_result.append(pred_idx[b][0].tolist())
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics_rec["count"] += 1

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.metrics_rec_ent = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "gate": 0, "count": 0,
                            'gate_count': 0}
        self.model.eval()
        # if is_test:
        #     val_dataset = dataset('data/train.json', self.opt)
        # else:
        #     val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_dataset=dataset4tgredial('data', self.opt,mode='test')
        val_set=CRSdataset4tgredial(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'],self.dict)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)

        for context,cur_context,response,entity_vec,entity_vector,movie,\
                concept_mask,cur_concept_mask,concept_vec,entity_rec,branch,m_branch,cur_l,leaf,i_rd\
                 in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                # add_edge_idx, add_edge_type = add_edge(entity_vector.view(-1).tolist())
                scores, preds, rec_scores, rec_loss,ent_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(
                    context.to(self.device), cur_context.to(self.device),response.to(self.device),
                    concept_mask.to(self.device),cur_concept_mask.to(self.device), seed_sets,entity_vector.to(self.device), concept_vec.to(self.device), entity_vec.to(self.device),
                    entity_rec.to(self.device),branch.to(self.device),m_branch.to(self.device),cur_l.to(self.device), leaf.to(self.device),test=True, maxlen=20, bsz=batch_size)
            # recs.extend(rec.cpu())
            #print(losses)
            #exit()
            self.metrics_cal_rec(rec_loss, rec_scores, movie)
            # self.metrics_cal_ent(rec_loss,rec_scores,entity_rec)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)
        # output_dict_rec = {key: self.metrics_rec_ent[key] / self.metrics_rec_ent['count'] for key in self.metrics_rec_ent}
        # print(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()



if __name__ == '__main__':
    seed_initial(22)

    args = setup_args4tgredial().parse_args()
    print(vars(args))
    if args.is_finetune == 'mov':
        loop = TrainLoop_fusion_rec4tgredial(vars(args), is_finetune='mov')
        # loop.model.load_model()
        loop.train()
    else:
        loop = TrainLoop_fusion_gen(vars(args), is_finetune='gen')
        # loop.train()
        # loop.model.load_model()
        # met = loop.val(True)
        loop.train()
    met = loop.val(True)
    # print(met)

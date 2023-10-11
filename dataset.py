import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
# from fuzzywuzzy import fuzz
import os
import random
# import spacy
# from entity_linker import match_nodes
# from data_pre_publish import preprocess
import argparse


MOVIE_TOKEN = '__MOVIE__'

class dataset4tgredial(object):
    def __init__(self,path,opt,mode='train'):
        random.seed(0)
        kg=json.load(open('tgredial_kg.json'))
        # self.movieid2entityid={int(i['MID']):int(i['global']) for i in kg['nodes'] if i['type']=='Movie'}
        #
        self.entity_max=len(kg['entity'])
        # self.token_dict={1:'__GENRE__',2:'__TIME__',3:'__ATTR__',4:'__SUBJECT__',5:'__PERSON__',6:'__MOVIE__'}
        # self.special_rd={'__GENRE__':1,'__TIME__':2,'__SUBJECT__':4,'__PERSON__':5,'__MOVIE__':6}
        # self.ent_token=['__GENRE__','__TIME__','__SUBJECT__','__PERSON__','__MOVIE__']
        # self.entity_mask=json.load(open('entity_mask.json'))
        #
        # self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))
        # self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        # self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']
        self.movie_max=opt['movie_max']
        # self.related_dic=self.get_concept_related_dic()
        # self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        # if mode=='train':
        #     data_=json.load(open(os.path.join(path,'train.json'),encoding='utf-8'))
        # elif mode=='test':
        #     data_=json.load(open(os.path.join(path,'test.json'),encoding='utf-8'))

        self.corpus = []
        # self.data=self.mask_entities(data_)
        #
        # with open('acl_{}1.json'.format(mode),'w') as f:
        #     json.dump(self.data,f)
        '''从这里一下全部注释了'''

        if 'train' in mode:
            self.data=json.load(open('tgredial/tgredial_train.json'))
            # self.data=json.load(open('data_with_MOVIE100.json'))
            # self.data=json.load(open('processed_data.json'))
            # d=json.load(open('processed_data_ent.json'))
        elif 'test' in mode:
            self.data=json.load(open('tgredial/tgredial_test.json'))
            # self.data=json.load(open('data_with_MOVIE100_test.json'))
            # self.data=json.load(open('processed_data_test_new.json'))
        # elif mode=='test':
        #     self.data=json.load(open('processed_data_test.json'))
        self.corpus=[]
        self.mode = mode
        if 'gen_test' == self.mode:
            self.rec_result = json.load(open('rec_result.json'))
        d=self.data
        if mode=='train':
            for i in range(len(d)):
                if i < len(d) - 1 and d[i]['user_id'] != d[i + 1]['user_id']:
                    self.corpus.extend(d[i]['context_tokens'])
                    self.corpus.append(d[i]['response'])
                elif i == len(d) - 1:
                    self.corpus.extend(d[i]['context_tokens'])
                    self.corpus.append(d[i]['response'])
            self.prepare_word2vec()
        self.word2index = json.load(open('word2index_tgredial.json', encoding='utf-8'))
        self.key2index = json.load(open('tgredial/word2id.json'))
        # self.key2index=json.load(open('key2index_3rd.json',encoding='utf-8'))

        # self.con2word=[0]*(len(self.key2index)+1)
        # for k,v in self.key2index.items():
        #     self.con2word[v]=self.word2index.get(k,0)
        #
        # self.n_kind_set={v:[] for k,v in self.token_dict.items()}
        # for k in self.n_kind_set.keys():
        #     for word in self.word2index.keys():
        #         if k in word:
        #             try:
        #                 self.n_kind_set[k].append(self.word2index[word])
        #             except:
        #                 pass

        # self.ENTTOKEN_rd=self.word2index.get('__ENT__',0)



        # self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        # self.response_text = []
        #
        # context_before=None
        # if mode=='test' and opt['is_finetune']=='gen':
        #     for data in self.data:
        #         if data['contexts'] == context_before:
        #             continue
        #         self.response_text.append(data['response_text'])
        #         context_before=data['contexts']

        # self.ent_token=self.word2index['__ENT__']




        #self.co_occurance_ext(self.data)
        #exit()


    def mask_entities(self,dataset):
        cases=[]
        total_rd=-1
        for data in tqdm(dataset):
            total_rd+=1
            # if total_rd==50:
            #     with open('short_processed_data.json','w') as f:
            #         json.dump(cases,f)
            #     raise ValueError
            history=data['context']
            response=data['utterance']
            dialog_num=data['dialog_num']
            entity=data['mentioned']
            entity_label=data['new_mentioned']

            token_text_all=[]
            unmasked_text_all=[]
            match_list=[]
            for sen in history:
                token_text=word_tokenize(sen)

                token_text_com=[]
                num=0
                while num<len(token_text):
                    if token_text[num]=='@' and num+1<len(token_text):
                        try:
                            movie_token = token_text[num] + token_text[num + 1]
                            token_text_com.append('__MOVIE__'+str(self.movieid2entityid(int(token_text[num+1]))))
                            num += 2
                        except:
                            token_text_com.append(token_text[num])
                            num += 1
                    else:
                        token_text_com.append(token_text[num])
                        num+=1
                # if total_rd>100:
                #     print(sen)
                #     print(token_text_com)
                unmasked_text_all.append(deepcopy(token_text_com))
                match_sublist, sen = match_nodes(token_text_com, entity)
                match_list.extend(match_sublist)
                token_text_com = word_tokenize(sen)
                # if total_rd>100:
                #     print(token_text_com)
                #     print('_________')
                # if total_rd==105:
                #     raise ValueError
                token_text_all.append(token_text_com)

            token_response=word_tokenize(response)
            token_res_com = []
            token_res_text=[]
            num = 0
            movie_rec=[]
            mentioned_list=[]
            while num < len(token_response):
                if token_response[num] == '@' and num + 1 < len(token_response):
                    # token_text_com.append(token_text[num]+token_text[num+1])
                    try:
                        movie_token = token_response[num] + token_response[num + 1]
                        # token_res_com.append(movie_token)
                        token_res_text.append(movie_token)
                        token_res_com.append('__MOVIE__'+str(self.movieid2entityid(int(token_response[num+1]))))
                        movie_rec.append(self.movieid2entityid[int(token_response[num+1])])
                        num += 2
                    except:
                        token_res_com.append(token_response[num])
                        token_res_text.append(token_response[num])
                        num += 1
                else:
                    token_res_com.append(token_response[num])
                    token_res_text.append(token_response[num])
                    num += 1
                mentioned_sublist, response = match_nodes(token_res_com, entity_label)
                mentioned_list.extend(mentioned_sublist)
                token_res_com = word_tokenize(response)
            self.corpus.extend(token_text_all)
            self.corpus.append(token_res_com)
            if len(token_text_all)==0:
                continue
            if len(movie_rec)!=0:
                rec=1
                '''train 不需要让每一个例子只有一个需要推荐的电影  这样反而不利于训练， 故在此处测试集和训练集分别处理'''
                for movie in movie_rec:
                    cases.append({'contexts': deepcopy(token_text_all),'unmasked_contexts': deepcopy(unmasked_text_all), 'response': deepcopy(token_res_com),'response_text':deepcopy(token_res_text), 'entity': deepcopy(entity),
                                  'movie': movie,'rec': rec,'mentioned':mentioned_list,'entity_label':entity_label,'movie_rec':movie_rec,'dialog_num':dialog_num,'ent_list':match_list})
            else:
                rec = 0
                cases.append({'contexts': deepcopy(token_text_all),'unmasked_contexts': deepcopy(unmasked_text_all), 'response': deepcopy(token_res_com), 'response_text':deepcopy(token_res_text),'entity': deepcopy(entity),
                  'movie': self.entity_max, 'rec': rec,'mentioned':mentioned_list,'entity_label':entity_label,'movie_rec':movie_rec,'dialog_num':dialog_num,'ent_list':match_list})

        return cases

    def get_concept_related_dic(self):
        related_dic={}
        node2index = json.load(open('key2index_3rd.json', encoding='utf-8'))
        f = open('conceptnet_edges2nd.txt', encoding='utf-8')
        stopwords = set([word.strip() for word in open('stopwords.txt', encoding='utf-8')])
        for line in f:
            lines = line.strip().split('\t')
            entity0 = node2index[lines[1].split('/')[0]]
            entity1 = node2index[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                continue
            if lines[0] in ['NotHasProperty', 'NotDesires', 'Antonym', 'NotCapableOf', 'DistinctFrom']:
                pass
            else:
                if entity0 in related_dic:
                    related_dic[entity0].append(entity1)
                else:
                    related_dic[entity0]=[entity1]
                if entity1 in related_dic:
                    related_dic[entity1].append(entity0)
                else:
                    related_dic[entity1]=[entity0]
        return related_dic

    def prepare_word2vec(self):
        import gensim
        model=gensim.models.word2vec.Word2Vec(self.corpus,vector_size=300,min_count=1)
        model.save('word2vec_redial')
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}
        #word2index['_split_']=len(word2index)+4
        #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = [[0] * 300] * 4 + [model.wv[word] for word in word2index]+[[0]*300]
        import numpy as np

        word2index['_split_']=len(word2index)+4
        json.dump(word2index, open('word2index_tgredial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        print(np.shape(word2embedding))
        np.save('word2vec_tgredial.npy', word2embedding)
    # def prepare_word2vec(self):
    #     import gensim
    #     # model=gensim.models.word2vec.Word2Vec(self.corpus,vector_size=300,min_count=1)
    #     model=gensim.models.word2vec.Word2Vec(self.corpus,size=300,min_count=1)
    #     # model.save('word2vec_redial')
    #     # word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}
    #     word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
    #     #word2index['_split_']=len(word2index)+4
    #     #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
    #     word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index]+[[0]*300]
    #     import numpy as np
    #
    #     word2index['_split_']=len(word2index)+4
    #     print('---Saving word2vec data... ----')
    #     json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
    #
    #     print(np.shape(word2embedding))
    #     np.save('word2vec_redial.npy', word2embedding)

    def padding_w2v(self,sentence,unmasked,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        special_mask=[]
        related_concept=[]
        for word,unword in zip(sentence,unmasked):
            # if word == self.ent_token:
            #     vector.append(self.word2index.get(word))
            #     special_mask.append(self.special_rd[word])
            # else:
            vector.append(self.word2index.get(word,unk))
            special_mask.append(0)
            #if word.lower() not in self.stopwords:
            concept_rd=self.key2index.get(word.lower(),0)
            # if concept_rd!=0 and concept_rd in self.related_dic:
            #     related_concept.extend(self.related_dic[concept_rd])
            concept_mask.append(concept_rd)
            #else:
            #    concept_mask.append(0)
            if '@' in word:
                try:
                    id=self.movieid2entityid[int(word[1:])]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)
        special_mask.append(0)


        related_concept_set=set(related_concept)
        cut_set=set(concept_mask)
        related_concept=list(related_concept_set-cut_set)

        if len(related_concept)>50:
            related_concept=random.sample(related_concept,50)
        elif len(related_concept)<50:
            related_concept+=[0]*(50-len(related_concept))


        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:],special_mask[-max_length:],related_concept
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length],special_mask[-max_length:],related_concept
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max],special_mask+(max_length-len(vector))*[0],related_concept

    def padding_context(self,contexts,unmasked,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            contexts_com=[]
            unmasked_com=[]
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            for sen in unmasked[-self.max_count:-1]:
                unmasked_com.extend(sen)
                unmasked_com.append('_split_')
            _,_,concept_mask,_,_,_ = self.padding_w2v(contexts_com,unmasked_com,self.max_c_length,transformer)
            _, _, cur_concept_mask, _, _, _ = self.padding_w2v(contexts[-1],contexts[-1], self.max_c_length, transformer)
            contexts_com.extend(contexts[-1])
            unmasked_com.extend(contexts[-1])
            vec,v_l,_,dbpedia_mask,special_mask,related_concept=self.padding_w2v(contexts_com,unmasked_com,self.max_c_length,transformer)

            return vec,v_l,concept_mask,dbpedia_mask,special_mask,related_concept,cur_concept_mask

    def padding_context1(self,contexts,unmasked,ent_seq,kw,cur_kw,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            contexts_com=[]
            unmasked_com=[]
            i=0
            for cur_l in reversed(ent_seq):
                if cur_l != []:
                    break
                i+=1

            for sen in contexts[-self.max_count:-1-i]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            for sen in unmasked[-self.max_count:-1-i]:
                unmasked_com.extend(sen)
                unmasked_com.append('_split_')
            vec,_,concept_mask,_,_,_ = self.padding_w2v(contexts_com,unmasked_com,self.max_c_length,transformer)
            cur_contexts_com = []
            cur_unmasked_com = []
            for sen in contexts[-1-i:]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
                cur_contexts_com.extend(sen)
                cur_contexts_com.append('_split_')
            for sen in unmasked[-1-i:]:
                unmasked_com.extend(sen)
                unmasked_com.append('_split_')
                cur_unmasked_com.extend(sen)
                cur_unmasked_com.append('_split_')
            cur_vec, _, cur_concept_mask, _, _, _ = self.padding_w2v(cur_contexts_com,cur_unmasked_com,self.max_c_length, transformer)

            _,v_l,_,dbpedia_mask,special_mask,related_concept=self.padding_w2v(contexts_com,unmasked_com,self.max_c_length,transformer)
            # concept_mask = [self.key2index[w.lower()] for w in kw if w.lower() in self.key2index]
            # cur_concept_mask = [self.key2index[w.lower()] for w in cur_kw if w.lower() in self.key2index]
            # concept_len = 256
            # if len(concept_mask)>concept_len:
            #     concept_mask = concept_mask[-concept_len:]
            # else:
            #     concept_mask = concept_mask + [0]*(concept_len - len(concept_mask))
            # cur_concept_len=32
            # if len(cur_concept_mask)>cur_concept_len:
            #     cur_concept_mask = cur_concept_mask[-cur_concept_len:]
            # else:
            #     cur_concept_mask = cur_concept_mask + [0]*(cur_concept_len - len(cur_concept_mask))

            return vec,cur_vec,v_l,concept_mask,dbpedia_mask,special_mask,related_concept,cur_concept_mask

    def gen_padding_context1(self,contexts,unmasked,ent_seq,rel_ent,pad=0,transformer=True):

        contexts_com=[]
        unmasked_com=[]
        i=0
        assert len(contexts) == len(ent_seq)

        for sen,entl in zip(contexts[:-1],ent_seq[:-1]):
            flag=0
            for ent in entl:
                if ent in rel_ent:
                    flag=1
                    break
            if flag==0:
                continue
            contexts_com.extend(sen)
            contexts_com.append('_split_')
        for sen in unmasked[:-1]:
            unmasked_com.extend(sen)
            unmasked_com.append('_split_')
        contexts_com.extend(contexts[-1])
        vec,v_l,concept_mask,dbpedia_mask,special_mask,related_concept=self.padding_w2v(contexts_com,unmasked_com,self.max_c_length,transformer)
        # concept_mask = [self.key2index[w.lower()] for w in kw if w.lower() in self.key2index]
        # cur_concept_mask = [self.key2index[w.lower()] for w in cur_kw if w.lower() in self.key2index]
        # concept_len = 256
        # if len(concept_mask)>concept_len:
        #     concept_mask = concept_mask[-concept_len:]
        # else:
        #     concept_mask = concept_mask + [0]*(concept_len - len(concept_mask))
        # cur_concept_len=32
        # if len(cur_concept_mask)>cur_concept_len:
        #     cur_concept_mask = cur_concept_mask[-cur_concept_len:]
        # else:
        #     cur_concept_mask = cur_concept_mask + [0]*(cur_concept_len - len(cur_concept_mask))

        return vec,v_l,concept_mask,dbpedia_mask,special_mask,related_concept

    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def pad_vector(self,vector,max_length,pad=0):
        if len(vector) > max_length:
            return vector[-max_length:]
        else:
            return vector+(max_length-len(vector))*[pad]
    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for i_rd,line in enumerate(self.data):
            if is_finetune and line['context_tokens'] == context_before:
                continue
            else:
                context_before = line['context_tokens']
            # context,c_lengths,concept_mask,dbpedia_mask,special_mask_con,related_concept,cur_concept_mask=self.padding_context(line['token_contexts'],line['context'])
            context, cur_context,c_lengths, concept_mask, dbpedia_mask, special_mask_con, related_concept, cur_concept_mask = \
                self.padding_context1(line['context_tokens'], line['context_tokens'],line['ent_seq'],0,0)
            response,r_length,concept_label_mask,_,special_mask_res,res_related_concept=self.padding_w2v(line['response'],line['response'],self.max_r_length)
            if False:
                mask_response,mask_r_length,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length
            assert len(special_mask_con)==self.max_c_length

            c_lengths, r_length, mask_r_length, concept_label_mask, dbpedia_mask, rec, special_mask_res, related_concept, ent_pos, ent_label = \
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            # context = []
            # for subsen in line['context_tokens']:
            #     context += subsen + [self.word2index['_split_']]
            # context = self.pad_vector(context,self.max_c_length)
            # cur_context = deepcopy(context)
            # response = self.pad_vector(line['response'],self.max_r_length)
            # mask_response = deepcopy(response)

            concept_mask = line['context_words']
            if len(concept_mask) < 128:
                concept_mask = concept_mask + [0]*(128-len(concept_mask))
            else:
                print(len(concept_mask))
                concept_mask = concept_mask[-128:]
            cur_concept_mask = line['word_seq'][-1]
            if len(cur_concept_mask) < 32:
                cur_concept_mask = cur_concept_mask + [0]*(32-len(cur_concept_mask))
            else:
                print(len(cur_concept_mask))
                cur_concept_mask = cur_concept_mask[-32:]
            entity = line['context_entities']


            if self.mode=='test':
                for movie in line['movie_rec']:
                    data_set.append([np.array(context),np.array(cur_context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,entity,
                                 movie,concept_mask,cur_concept_mask,concept_label_mask,dbpedia_mask,rec,entity,line['ent_rec'],line['movie_rec'],special_mask_res,related_concept,ent_pos,ent_label,\
                                     line['g_sub'],line['head_to_tail'],line['ent_seq'],i_rd])
            else:
                if len(line['ent_rec'])!=0:
                    data_set.append(
                    [np.array(context), np.array(cur_context), c_lengths, np.array(response), r_length, np.array(mask_response), mask_r_length,
                     entity,self.entity_max, concept_mask, cur_concept_mask, concept_label_mask, dbpedia_mask,rec,
                     entity, line['ent_rec'], line['movie_rec'], special_mask_res, related_concept,
                     ent_pos, ent_label, line['g_sub'], line['head_to_tail'], line['ent_seq'], i_rd])

        return data_set

    def gen_data_process(self, is_finetune=False):
        data_set = []
        context_before = []
        rec_result = []
        rrd=0
        if 'test' in self.mode:
            for line in self.data:
                movie_len = len(line['movie_rec'])
                if movie_len == 1:
                    rec_result.append(self.rec_result[rrd])
                    rrd += 1
                elif movie_len == 0:
                    rec_result.append(self.entity_num)
                else:
                    ele = list(set(self.rec_result[rrd:rrd + movie_len]))
                    assert len(ele) == 1
                    rec_result.append(ele[0])
                    rrd += movie_len

            assert len(rec_result) == len(self.data)
            for i_rd, couple in enumerate(zip(self.data,rec_result)):
                line,aim = couple
                # if len(line['contexts'])>2:
                #    continue
                if line['token_contexts'] == context_before:
                    continue
                else:
                    context_before = line['token_contexts']

                rel_ent = []
                for path in line['head_to_tail']:
                    if aim in path:
                        rel_ent.extend(path)
                rel_ent = list(set(rel_ent))

                context, c_lengths, concept_mask, dbpedia_mask, special_mask_con, related_concept= \
                    self.gen_padding_context1(line['token_contexts'], line['context'], line['ent_seq'],rel_ent)
                response, r_length, concept_label_mask, _, special_mask_res, res_related_concept = self.padding_w2v(
                    line['response'], line['response'], self.max_r_length)

                data_set.append([np.array(context),np.array(response), rel_ent, concept_mask])

        else:
            for i_rd, line in enumerate(self.data):
                # if len(line['contexts'])>2:
                #    continue
                if line['token_contexts'] == context_before:
                    continue
                else:
                    context_before = line['token_contexts']

                aim = line['movie_rec']
                rel_ent = []
                for mov in aim:
                    for path in line['head_to_tail']:
                        if mov in path:
                            rel_ent.extend(path)
                rel_ent = list(set(rel_ent))

                context,  c_lengths, concept_mask, dbpedia_mask, special_mask_con, related_concept = \
                    self.gen_padding_context1(line['token_contexts'], line['context'], line['ent_seq'],rel_ent)
                response, r_length, concept_label_mask, _, special_mask_res, res_related_concept = self.padding_w2v(
                    line['response'], line['response'], self.max_r_length)
                data_set.append([np.array(context), np.array(response), rel_ent, concept_mask])
                # mask_response, mask_r_length = response, r_length
                # ent_pos, ent_label = 0, 0
                # data_set.append(
                #     [np.array(context), np.array(cur_context), c_lengths, np.array(response), r_length,
                #      np.array(mask_response), mask_r_length,
                #      line['entity'], self.entity_max, concept_mask, cur_concept_mask, concept_label_mask,
                #      dbpedia_mask, line['rec'],
                #      line['entity'], line['entity_label'], line['movie_rec'], special_mask_res, related_concept,
                #      ent_pos, ent_label, line['g_sub'], line['head_to_tail'], line['ent_seq'], i_rd])


        return data_set


    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('co_occurance.txt','w',encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('movie_word.json','w',encoding='utf-8'),ensure_ascii=False)
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self,entities):
        return [self.entity2entityId[word] for word in entities]

class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num,dict):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1
        word2index = json.load(open('word2index_tgredial.json', encoding='utf-8'))
        self.stopwords = set([word2index[word.strip()] for word in open('stopwords.txt', encoding='utf-8') if word in word2index])
        self.word_num=len(dict)
        # self.ent_id=word2index['__ENT__']

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context,cur_context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, \
            concept_mask,cur_concept_mask,concept_label_mask,dbpedia_mask, rec,mentioned_list,entity_label,\
            movie_rec,special_mask_res,related_concept,ent_pos,ent_label,paths,h2t,ent_seq,i_rd = self.data[index]

        entity_vec = np.zeros(self.entity_num)

        # entity_vector=np.zeros(64,dtype=np.int)
        entity_vector = np.array([self.entity_num]*64)
        entity_label_vector = np.array([[self.entity_num]*20]*6, dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in movie_rec:
            if db!=0:
                db_vec[db]=1

        entity_label=[i for i in entity_label if (i<30452 or i>30458) and i!=30438]
        # entity_label = [i for i in entity_label if i < 6924]
        entity_rec=np.array(entity_label+[self.entity_num]*(32-len(entity_label)))
        if len(entity_rec)>32:
            print(len(entity_rec))
            raise ValueError

        # total_entity=np.array(entity+[self.entity_num]*(256-len(entity)))

        labels=np.zeros(self.entity_num)
        point=[0]*7
        # for label in entity_label:
        #     if label <= 6923:
        #         labels[label]=6
        #         entity_label_vector[5][point[6]]=label
        #         point[6]+=1
        #     elif label <= 19726:
        #         labels[label]=5
        #         entity_label_vector[4][point[5]] = label
        #         point[5] += 1
        #     elif label <= 30433:
        #         labels[label]=4
        #         entity_label_vector[3][point[4]] = label
        #         point[4] += 1
        #     elif label <= 30451:
        #         labels[label]=1
        #         entity_label_vector[0][point[1]] = label
        #         point[1] += 1
        #     elif label <= 30458:
        #         pass
        #     elif label <= 30470:
        #         labels[label]=2
        #         entity_label_vector[1][point[2]] = label
        #         point[2] += 1
        #     else:
        #         raise ValueError

        if list(labels)==[0]*self.entity_num:
            rec_mask=0
        else:
            rec_mask=1

        word_vec=np.zeros(self.word_num+4)
        word_set=list(set(context)-self.stopwords)
        for word in word_set:
            word_vec[word]=1

        cur_l = []
        for l in reversed(ent_seq):
            if l != []:
                cur_l = l
                break
        cur_l += [self.entity_num]*(16-len(cur_l))

        branch=[]
        m_branch=[]
        entity_=list(entity_vector)
        leaf = []
        for i in h2t:
            i = [ele for ele in i[1:] if ele not in cur_l]
            mi = [ele for ele in i[1:] if ele not in cur_l and ele<6924]
            if i!=[]:
                leaf.append(i[-1])
            m_branch.append(mi+[self.entity_num]*(32-len(mi)))
            branch.append(i+[self.entity_num]*(32-len(i)))
        branch += [[self.entity_num]*32 for _ in range(24-len(branch))]
        m_branch += [[self.entity_num] * 32 for _ in range(24 - len(m_branch))]

        leaf = [i for i in leaf if i not in cur_l and i >= 6924]
        leaf += [self.entity_num] * (32 - len(leaf))

        # i=0
        # movie_mentioned = []
        # for l in reversed(ent_seq):
        #     if l!=[] and i<7:
        #         i+=1
        #         continue
        #     movie_mentioned.extend([e for e in l if e<6924 and e in entity])
        # movie_mentioned += [self.entity_num]*(32-len(movie_mentioned))

        # cur_l=[]
        # for l in h2t:
        #     if l[-1] in cur_ent:
        #         assert len(l)<=16
        #         cur_l.append(l[1:]+[self.entity_num]*(17-len(l)))
        # assert len(cur_l) <= len(cur_ent)
        # cur_l += [[self.entity_num] * 16 for _ in range(14 - len(cur_l))]

        return np.array(context),np.array(cur_context),np.array(response), entity_vec,\
               entity_vector, movie, np.array(concept_mask),np.array(cur_concept_mask),concept_vec,\
               entity_rec,np.array(branch),np.array(m_branch),np.array(cur_l),np.array(leaf),i_rd

    def __len__(self):
        return len(self.data)

class CRSdataset4tgredial(Dataset):
    def __init__(self, dataset, entity_num, concept_num,dict):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1
        # word2index = json.load(open('word2index_tgredial.json', encoding='utf-8'))
        # self.stopwords = set([word2index[word.strip()] for word in open('stopwords.txt', encoding='utf-8') if word in word2index])
        # self.word_num=len(dict)
        # self.ent_id=word2index['__ENT__']

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context,cur_context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, \
            concept_mask,cur_concept_mask,concept_label_mask,dbpedia_mask, rec,mentioned_list,entity_label,\
            movie_rec,special_mask_res,related_concept,ent_pos,ent_label,paths,h2t,ent_seq,i_rd = self.data[index]

        entity_vec = np.zeros(self.entity_num)

        # entity_vector=np.zeros(64,dtype=np.int)
        entity_vector = np.array([self.entity_num]*64)
        entity_label_vector = np.array([[self.entity_num]*20]*6, dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in movie_rec:
            if db!=0:
                db_vec[db]=1

        entity_label=[i for i in entity_label if (i<30452 or i>30458) and i!=30438]
        # entity_label = [i for i in entity_label if i < 6924]
        entity_rec=np.array(entity_label+[self.entity_num]*(32-len(entity_label)))
        if len(entity_rec)>32:
            print(len(entity_rec))
            raise ValueError

        # total_entity=np.array(entity+[self.entity_num]*(256-len(entity)))

        labels=np.zeros(self.entity_num)
        point=[0]*7
        # for label in entity_label:
        #     if label <= 6923:
        #         labels[label]=6
        #         entity_label_vector[5][point[6]]=label
        #         point[6]+=1
        #     elif label <= 19726:
        #         labels[label]=5
        #         entity_label_vector[4][point[5]] = label
        #         point[5] += 1
        #     elif label <= 30433:
        #         labels[label]=4
        #         entity_label_vector[3][point[4]] = label
        #         point[4] += 1
        #     elif label <= 30451:
        #         labels[label]=1
        #         entity_label_vector[0][point[1]] = label
        #         point[1] += 1
        #     elif label <= 30458:
        #         pass
        #     elif label <= 30470:
        #         labels[label]=2
        #         entity_label_vector[1][point[2]] = label
        #         point[2] += 1
        #     else:
        #         raise ValueError

        if list(labels)==[0]*self.entity_num:
            rec_mask=0
        else:
            rec_mask=1

        # word_vec=np.zeros(self.word_num+4)
        # word_set=list(set(context)-self.stopwords)
        # for word in word_set:
        #     word_vec[word]=1

        cur_l = []
        for l in reversed(ent_seq):
            if l != []:
                cur_l = l
                break
        cur_l += [self.entity_num]*(16-len(cur_l))

        branch=[]
        m_branch=[]
        entity_=list(entity_vector)
        leaf = []
        for i in h2t:
            i = [ele for ele in i[1:] if ele not in cur_l]
            mi = [ele for ele in i[1:] if ele not in cur_l and ele<6924]
            if i!=[]:
                leaf.append(i[-1])
            m_branch.append(mi+[self.entity_num]*(32-len(mi)))
            branch.append(i+[self.entity_num]*(32-len(i)))
        branch += [[self.entity_num]*32 for _ in range(32-len(branch))]
        m_branch += [[self.entity_num] * 32 for _ in range(32 - len(m_branch))]

        leaf = [i for i in leaf if i not in cur_l and i >= 6924]
        leaf += [self.entity_num] * (32 - len(leaf))

        # i=0
        # movie_mentioned = []
        # for l in reversed(ent_seq):
        #     if l!=[] and i<7:
        #         i+=1
        #         continue
        #     movie_mentioned.extend([e for e in l if e<6924 and e in entity])
        # movie_mentioned += [self.entity_num]*(32-len(movie_mentioned))

        # cur_l=[]
        # for l in h2t:
        #     if l[-1] in cur_ent:
        #         assert len(l)<=16
        #         cur_l.append(l[1:]+[self.entity_num]*(17-len(l)))
        # assert len(cur_l) <= len(cur_ent)
        # cur_l += [[self.entity_num] * 16 for _ in range(14 - len(cur_l))]
        return np.array(context),np.array(cur_context),np.array(response), entity_vec,\
               entity_vector, movie, np.array(concept_mask),np.array(cur_concept_mask),concept_vec,\
               entity_rec,np.array(branch),np.array(m_branch),np.array(cur_l),np.array(leaf),i_rd

    def __len__(self):
        return len(self.data)

class gen_CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num,dict):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1
        word2index = json.load(open('word2index_redial.json', encoding='utf-8'))
        self.stopwords = set([word2index[word.strip()] for word in open('stopwords.txt', encoding='utf-8') if word in word2index])
        self.word_num=len(dict)
        # self.ent_id=word2index['__ENT__']

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        '''gengengengen'''
        context, response, rel_ent, concept_mask = self.data[index]
        rel_ent = rel_ent + [self.entity_num]*(64-len(rel_ent))
        # entity_vec = np.zeros(self.entity_num)
        #
        # # entity_vector=np.zeros(64,dtype=np.int)
        # entity_vector = np.array([self.entity_num]*64)
        # entity_label_vector = np.array([[self.entity_num]*20]*6, dtype=np.int)
        # point=0
        # for en in entity:
        #     entity_vec[en]=1
        #     entity_vector[point]=en
        #     point+=1
        #
        # concept_vec=np.zeros(self.concept_num)
        # for con in concept_mask:
        #     if con!=0:
        #         concept_vec[con]=1
        #
        # db_vec=np.zeros(self.entity_num)
        # for db in movie_rec:
        #     if db!=0:
        #         db_vec[db]=1
        #
        # entity_label=[i for i in entity_label if (i<30452 or i>30458) and i!=30438]
        # # entity_label = [i for i in entity_label if i < 6924]
        # entity_rec=np.array(entity_label+[self.entity_num]*(32-len(entity_label)))
        # if len(entity_rec)>32:
        #     print(len(entity_rec))
        #     raise ValueError
        #
        # # total_entity=np.array(entity+[self.entity_num]*(256-len(entity)))
        #
        # labels=np.zeros(self.entity_num)
        # point=[0]*7
        # for label in entity_label:
        #     if label <= 6923:
        #         labels[label]=6
        #         entity_label_vector[5][point[6]]=label
        #         point[6]+=1
        #     elif label <= 19726:
        #         labels[label]=5
        #         entity_label_vector[4][point[5]] = label
        #         point[5] += 1
        #     elif label <= 30433:
        #         labels[label]=4
        #         entity_label_vector[3][point[4]] = label
        #         point[4] += 1
        #     elif label <= 30451:
        #         labels[label]=1
        #         entity_label_vector[0][point[1]] = label
        #         point[1] += 1
        #     elif label <= 30458:
        #         pass
        #     elif label <= 30470:
        #         labels[label]=2
        #         entity_label_vector[1][point[2]] = label
        #         point[2] += 1
        #     else:
        #         raise ValueError
        #
        # if list(labels)==[0]*self.entity_num:
        #     rec_mask=0
        # else:
        #     rec_mask=1
        #
        # word_vec=np.zeros(self.word_num+4)
        # word_set=list(set(context)-self.stopwords)
        # for word in word_set:
        #     word_vec[word]=1
        #
        # cur_l = []
        # for l in reversed(ent_seq):
        #     if l != []:
        #         cur_l = l
        #         break
        # cur_l += [self.entity_num]*(16-len(cur_l))
        #
        # branch=[]
        # m_branch=[]
        # entity_=list(entity_vector)
        # leaf = []
        # for i in h2t:
        #     i = [ele for ele in i[1:] if ele not in cur_l]
        #     mi = [ele for ele in i[1:] if ele not in cur_l and ele<6924]
        #     if i!=[]:
        #         leaf.append(i[-1])
        #     m_branch.append(mi+[self.entity_num]*(32-len(mi)))
        #     branch.append(i+[self.entity_num]*(32-len(i)))
        # branch += [[self.entity_num]*32 for _ in range(24-len(branch))]
        # m_branch += [[self.entity_num] * 32 for _ in range(24 - len(m_branch))]
        #
        # leaf = [i for i in leaf if i not in cur_l and i >= 6924]
        # leaf += [self.entity_num] * (32 - len(leaf))

        # i=0
        # movie_mentioned = []
        # for l in reversed(ent_seq):
        #     if l!=[] and i<7:
        #         i+=1
        #         continue
        #     movie_mentioned.extend([e for e in l if e<6924 and e in entity])
        # movie_mentioned += [self.entity_num]*(32-len(movie_mentioned))

        # cur_l=[]
        # for l in h2t:
        #     if l[-1] in cur_ent:
        #         assert len(l)<=16
        #         cur_l.append(l[1:]+[self.entity_num]*(17-len(l)))
        # assert len(cur_l) <= len(cur_ent)
        # cur_l += [[self.entity_num] * 16 for _ in range(14 - len(cur_l))]
        return np.array(context),np.array(response), np.array(rel_ent),np.array(concept_mask)

    def __len__(self):
        return len(self.data)
# class CRSdataset(Dataset):
#     def __init__(self, dataset, entity_num, concept_num):
#         self.data=dataset
#         self.entity_num = entity_num
#         self.concept_num = concept_num+1
#
#     def __getitem__(self, index):
#         '''
#         movie_vec = np.zeros(self.entity_num, dtype=np.float)
#         context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
#         for en in movie:
#             movie_vec[en] = 1 / len(movie)
#         return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
#         '''
#         context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec, movies_gth, movies_num,new_person,new_subject,entity_list,entity_mask= self.data[index]
#         entity_vec = np.zeros(self.entity_num)
#         entity_vector=np.zeros(50,dtype=np.int)
#         point=0
#         for en in entity:
#             entity_vec[en]=1
#             entity_vector[point]=en
#             point+=1
#
#         concept_vec=np.zeros(self.concept_num)
#         for con in concept_mask:
#             if con!=0:
#                 concept_vec[con]=1
#
#         db_vec=np.zeros(self.entity_num)
#         for db in dbpedia_mask:
#             if db!=0:
#                 db_vec[db]=1
#
#         person_vec=np.zeros(self.entity_num)
#         subject_vec=np.zeros(self.entity_num)
#         for p in new_person:
#             person_vec[p]=1
#         for s in new_subject:
#             subject_vec[s]=1
#
#         return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec, movies_gth, movies_num,person_vec,subject_vec,np.array(entity_list),np.array(entity_mask)
#
#     def __len__(self):
#         return len(self.data)

# if __name__=='__main__':
#     ds=dataset('data/train_data.jsonl')
#     print()
'''
0-6923 movie
6924-19726 person
19727-30433 subject
30434-30451 genre
30452-30458 attr
30459-30470 time
'''
def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=32)
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
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
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

    train.add_argument("-n_entity","--n_entity",type=int,default=30472)
    # train.add_argument("-n_relation","--n_relation",type=int,default=214)
    train.add_argument("-n_concept","--n_concept",type=int,default=29308)
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

if __name__ == '__main__':
    args = setup_args().parse_args()
    b = dataset('data', vars(args), mode='test', process_raw_data=True)
    a=dataset('data',vars(args),mode='train',process_raw_data=True)



    # spe_token=['__GENRE__','__SUBJECT__','__PERSON__','__TIME__']
    # data=json.load(open('acl_test.json'))
    # final=[]
    # for line in tqdm(data):
    #     contexts=[]
    #     for sen in line['contexts']:
    #         contexts.append(['__ENT__' if word in spe_token else word for word in sen ])
    #     res=['__ENT__' if word in spe_token else word for word in line['response']]
    #     # final.append({'contexts':contexts,'response':res,'dialog_num':line['dialog_num']})
    #     line['contexts']=contexts
    #     line['response']=res
    #     final.append(line)
    # with open('acl_test_ent.json','w') as f:
    #     json.dump(final,f)


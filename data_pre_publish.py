import json
import sys
import torch
import json
import torch.nn as nn
import random
import os.path as osp
import os
import pickle as pkl
from tqdm import tqdm
import spacy
import numpy as np
from fuzzywuzzy import fuzz

model = "en_core_web_sm"
# model = "en"
print('spacy loading', model)
nlp = spacy.load(model)

dataset_name = "redial"

# root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# path = osp.join(root, "data", "redial")
# global_graph_path = osp.join(path, "raw", 'redial_kg.json')
# match_path = osp.join(path, "raw", 'match_results')
# save_path = osp.join(path, "raw", 'match_results_new')
f = open('data/redial_kg.json')
graph = json.load(f)
nodes = graph["nodes"]
relations = graph["relations"]

print(len(nodes))
# print(len(relations))

person_dict = {}
subject_dict = {}
genre_dict = {}
general_dict = {}
movie_dict = {}
time_dict = {}

#genre_match = {"Action": ["action"], "Adventure": ["adventure"],
#                "Animation": ['anime', 'animation', 'animated', 'cartoon'], "Children": ['child', 'kids', 'children'],
#                "Comedy": ['comedy', 'funny', 'humor', 'comedies'], "Crime": ['crime'],
#                "Documentary": ['documentary', 'documentaries'], "Drama": ['drama'], "Fantasy": ['fantasy', 'fantasies'],
#                "Film-Noir": ['film noir', 'neo noir'], "Horror": ['horror', 'scary'],
#                "Musical": ['musical', 'musician'], "Mystery": ['mystery', 'mysteries'],
#                "Romance": ['romance', 'romantic'], "Sci-Fi": ['sci fi', 'sci-fi', 'science fiction', 'fiction'],
#                "Thriller": ['thriller', 'thrilling'], "War": ['war'], "Western": ['western']}
genre_match = {"Action": ["action"], "Adventure": ["adventure"],
               "Animation": ['anime', 'animation', 'animated', 'cartoon'], "Children": ['child', 'kids', 'children'],
               "Comedy": ['comedy','humor', 'comedies'], "Crime": ['crime'],
               "Documentary": ['documentary', 'documentaries'], "Drama": ['drama'], "Fantasy": ['fantasy', 'fantasies'],
               "Film-Noir": ['film noir', 'neo noir'], "Horror": ['horror', 'scary'],
               "Musical": ['musical', 'musician'], "Mystery": ['mystery', 'mysteries'],
               "Romance": ['romance', 'romantic'], "Sci-Fi": ['sci fi', 'sci-fi', 'science fiction', 'fiction'],
               "Thriller": ['thriller', 'thrilling'], "War": ['war'], "Western": ['western']}

general_match = {"Movie": [], "Actor": ["actor", "actress"], "Director": ["director"],
                 "Genre": ["genre", "type of movie", "kind of movie"],
                 "Time": ["time", "era", "decade", "older", "newer"], "Attr": [], "Subject": ["subject"], "None": []}
time_match = {"1900": ["190"], "1910": ["191"], "1920": ["192", "20s", "20's"], "1930": ["193", "30s", "30's"],
              "1940": ["194", "40s", "40's"], "1950": ["195", "50s", "50's"], "1960": ["196", "60s", "60's"],
              "1970": ["197", "70s", "70's"], "1980": ["198", "80s", "80's"], "1990": ["199", "90s", "90's"],
              "2000": ['200'], "2010": ['201'], "2020": ['202']}

for i, node in enumerate(nodes):
    if node['type'] == "Movie":
        idx = int(node['MID'])
        movie_dict[idx] = i
    if node['type'] == "Person":
        name = node['name'].replace("_", " ").lower()
        person_dict[name] = i
    if node['type'] == "Subject":
        name = node['name'].replace("_", " ").lower()
        subject_dict[name] = i
    if node['type'] == "Genre":
        name = node['name']
        genre_dict[name] = i
    if node['type'] == "Attr":
        name = node['name']
        general_dict[name] = i
    if node['type'] == "Time":
        name = node['name'].lower().replace("s", "")
        time_dict[name] = i


def match_movie(utterance):
    def get_idx(tok):
        nums = "0123456789"
        result = ""
        for s in tok:
            if s in nums:
                result += s
        try:
            return int(result)
        except:
            return ""

    results = []
    refined_utterance = ""

    if dataset_name == "redial":
        tokens = utterance.split()
        for token in tokens:
            if "@" in token:
                idx = get_idx(token)
                if idx != "" and idx in movie_dict.keys():
                    results.append(movie_dict[idx])
                refined_utterance += "MOV" + " "
            else:
                refined_utterance += token + " "
    else:
        if "RECOMMEND" in utterance:
            tokens = utterance.split()
            for i, token in enumerate(tokens):
                if "MID" in token and tokens[i - 1] == "RECOMMEND":
                    idx = get_idx(token)
                    if idx != "" and idx in movie_dict.keys():
                        results.append(movie_dict[idx])
    return results, refined_utterance


def fuzzy_match_person(name):
    for item in person_dict.keys():
        score = fuzz.partial_ratio(item.replace("_", " "), name.lower())
        if score > 85:
            return item
    return None


def fuzzy_match_subject(name, threshold1=80, threshold2=90):
    if len(name) > 8 or name == "Marvel" or name == "Netflix" or name == "Disney" or name == "Pixar":
        for item in subject_dict.keys():
            score = fuzz.ratio(item.replace("_", " "), name.lower())
            score_part = fuzz.partial_ratio(item.replace("_", " "), name.lower())
            if score > threshold1 or score_part > threshold2:
                return item
    elif len(name) < 5:
        if name == "Love":
            return None
        for item in subject_dict.keys():
            score = fuzz.ratio(item.replace("_", " "), name.lower())
            if score > 95:
                return item
    else:
        for item in subject_dict.keys():
            score = fuzz.ratio(item.replace("_", " "), name.lower())
            if score > threshold1:
                return item

    return None


def fuzzy_match_general(utter):
    match_list = []
    for key, value in general_dict.items():
        for item in general_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list


def fuzzy_match_genre(utter):
    match_list = []
    for key, value in genre_dict.items():
        for item in genre_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list


def fuzzy_match_time(utter):
    match_list = []
    for key, value in time_dict.items():
        for item in time_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list


def fuzzy_match_mentioned(name, mentioned):
    # print(colored(name,"blue"))
    for item in mentioned:
        # print(nodes[item]['name'])
        score = fuzz.partial_ratio(name.lower(), nodes[item]['name'].replace("_", " ").lower())
        # print(score)
        if score > 90:
            return item
    # input()
    return None


# ProRec Template Entity Linker:
# 1.regular expression to match General + Genre
# 2.NER from Spacy to locate person entities
# 3.el by Fuzzywuzzy for full name in person set
# 4.el by Fuzzywuzzy for partial name in the mentioned set

def match_nodes(utterance, mentioned):
    matched = []

    movie, refined = match_movie(utterance)
    matched = matched + movie

    general = fuzzy_match_general(refined)
    matched = matched + general

    genre = fuzzy_match_genre(refined)
    matched = matched + genre

    time = fuzzy_match_time(refined)
    matched = matched + time

    with nlp.disable_pipes("tagger", "parser"):
        tgt = nlp(utterance)
        for ent in tgt.ents:
            if ent.label_ == "PERSON" or ent.label_ == "ORG":
                if ent.text.lower() in person_dict.keys():
                    matched.append(person_dict[ent.text.lower()])
                    continue
                if ent.text.lower() in subject_dict.keys():
                    matched.append(subject_dict[ent.text.lower()])
                    continue

                if " " in ent.text:
                    key = fuzzy_match_person(ent.text)
                    if key != None:
                        matched.append(person_dict[key])
                        continue
                key = fuzzy_match_mentioned(ent.text, mentioned)
                if key != None:
                    matched.append(key)
                    continue
                key = fuzzy_match_subject(ent.text)
                if key != None:
                    matched.append(subject_dict[key])
    return matched

global neigh
def merge_path(ent,paths):

    for i,path in enumerate(paths):
        for ii,node in enumerate(reversed(path)):
            if node==-1:
                n_path=[-1,ent]
                paths.append(n_path)
                break
            elif ent in neigh[node]:
                n_path=path[:-1-ii]+[ent]
                paths.append(n_path)
                break

    return paths

kg=json.load(open('data/redial_kg.json'))
def extract_entity_sequence(dataset):
    neigh={}
    for e in kg['relations']:
        if e[0] in neigh:
            neigh[e[0]].append(e[1])
        else:
            neigh[e[0]] = [e[1]]
        if e[1] in neigh:
            neigh[e[1]].append(e[0])
        else:
            neigh[e[1]] = [e[0]]

    # dataset=json.load(open('data/test.json'))
    paths=[[-1]]
    new_dataset=[]
    for data in tqdm(dataset):
        ent_sequential_list=[]
        for sen in data['context']:
            cur_ent=match_nodes(sen,data['entity'])
            ent_sequential_list.append(cur_ent)
            # for ent in cur_ent:
            #     paths=merge_path(ent,paths)
        cur_ent=match_nodes(data['utterance'],data['entity_label'])
        ent_sequential_list.append(cur_ent)
        assert len(ent_sequential_list)==len(data['context'])+1
        data['ent_seq']=ent_sequential_list
        new_dataset.append(data)

    # with open('data/test_new.json','w') as f:
    #     json.dump(new_dataset,f)
    return new_dataset

def process_all_pkl():
    all = pkl.load(open('tgredial/all_data.pkl','rb'))
    with open('word2index_tgredial.json','w') as f:
        json.dump(all[4]['tok2ind'],f)
    with open('tgredial_kg.json', 'w') as f:
        json.dump(all[3]['entity_kg'], f)
    with open('tgredial/tgredial_train_acl.json', 'w') as f:
        json.dump(all[0], f)
    with open('tgredial/tgredial_test_acl.json', 'w') as f:
        json.dump(all[2], f)
    with open('tgredial_word_kg.json', 'w') as f:
        json.dump(all[3]['word_kg'], f)
    np.save('word2vec_tgredial.npy',all[3]['embedding'])


def preprocess(raw_data):
    dataset = extract_entity_sequence(raw_data)
    new_data=[]
    import networkx as nx
    def is_valid(e):
        '''30452-30458 attr'''
        d = (e>=30452) and (e<=30458)
        return (not d)
    nodes=[i['global'] for i in kg['nodes']]
    edges=[(e[0],e[1]) for e in kg['relations'] if is_valid(e[0]) and is_valid(e[1])]
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # dataset=json.load(open('data/test_new.json'))
    # dataset =  json.load(open('../DealCRS/train_acl.json'))

    i=0
    for data in tqdm(dataset):
        i+=1
        # if i<2:
        #     continue
        # if i==30:
        #     with open('short_processed_data.json','w') as f:
        #         json.dump(new_data,f)
        #     raise ValueError
        '''delete entities of responses'''
        # data['ent_sequential_list']=data['ent_sequential_list'][:-1]
        ent_slist=data['ent_seq'][:-1]
        ent_seq_new=[]
        for l in ent_slist:
            new_l=[i for i in l if i in data['entity']]
            ent_seq_new.append(new_l)
        data['ent_seq']=ent_seq_new

        g=nx.Graph()
        g.add_node(-1)
        last_list=[]

        ent_whole=[]
        for ent_list in data['ent_seq']:
            ent_whole.extend(ent_list)
        for ii,cur_ent in enumerate(ent_whole):
            if cur_ent>=30452 and cur_ent<=30458:
                continue
            if cur_ent in g.nodes:
                continue
            g.add_node(cur_ent)

            for rd in range(ii-1,-2,-1):
                if rd==-1:
                    g_node=-1
                else:
                    g_node=ent_whole[rd]
                flag = 0

                if g_node==-1 or g_node==cur_ent:
                    flag=1
                    if g_node==-1:
                        g.add_edge(-1, cur_ent)
                    continue
                if nx.has_path(G,g_node,cur_ent) and nx.shortest_path_length(G,g_node,cur_ent)<=1:
                    flag=1
                    g.add_edge(g_node,cur_ent)
                if flag==1:
                    break

        last_list=[i[0] for i in g.degree if i[1]==1 and i[0]!=-1]

        records=[]
        for ent in last_list:
            llist=list(nx.all_simple_paths(g,-1,ent))
            assert len(llist)==1 or len(llist)==0
            if len(llist)==0:
                continue
            records.append(llist[0])

        data['head_to_tail']=records
        # new_data.append(data)
        data['g_nodes']=list(g.nodes)
        data['g_edges']=list(g.edges)
        g = g.to_undirected()
        g.remove_node(-1)
        data['g_sub'] = [list(i) for i in list(nx.connected_components(g))]
        new_data.append(data)

    # with open('data/test_acl.json','w') as f:
    #     json.dump(new_data,f)
    # with open('../DealCRS/train_acl1.json','w') as f:
    #     json.dump(new_data,f)
    return new_data









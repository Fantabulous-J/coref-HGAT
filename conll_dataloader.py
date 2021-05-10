import json
import random
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from transformers import BertTokenizer
import dgl

import util


class CoNLLCorefResolution(object):
    def __init__(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                 cluster_ids, sentence_map, subtoken_map, srl_dict):
        self.doc_key = doc_key
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.text_len = text_len
        self.speaker_ids = speaker_ids
        self.genre = genre
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.cluster_ids = cluster_ids
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map
        self.srl_dict = srl_dict


class CoNLLDataset(Dataset):
    def __init__(self, features: List[CoNLLCorefResolution], tokenizer: BertTokenizer, config, dependencies,
                 dep_tag2id, sign="train"):
        self.features = features
        self.config = config
        self.sign = sign
        self.tokenizer = tokenizer
        self.dependencies = dependencies
        self.dep_tag2id = dep_tag2id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        example = (feature.doc_key, feature.input_ids, feature.input_mask, feature.text_len, feature.speaker_ids,
                   feature.genre, feature.gold_starts, feature.gold_ends, feature.cluster_ids, feature.sentence_map,
                   feature.subtoken_map, feature.srl_dict)
        if self.sign == 'train' and len(example[1]) > self.config["max_training_sentences"]:
            example = truncate_example(*example, self.config)

        example = self.create_graph(*example, self.dependencies[item])
        return example

    def create_graph(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, srl_dict, dependency):
        graph = dgl.graph([])
        graph.set_n_initializer(dgl.init.zero_initializer)
        sentence_start_idx = sentence_map[0]
        sentence_end_idx = sentence_map[-1]
        num_tokens = sentence_map.size()[0]
        token_range = torch.arange(0, num_tokens, dtype=torch.int64)

        # dependency parsing trees (token -> token)
        dependency = dependency[sentence_start_idx: sentence_end_idx + 1]
        graph.add_nodes(num_tokens)
        graph.ndata['unit'] = torch.zeros(num_tokens)
        graph.ndata['dtype'] = torch.zeros(num_tokens)
        root_token_ids = None
        for sent_dep in dependency:
            for deprel, head_id, word_id in sent_dep:
                token_ids = token_range[subtoken_map == (word_id - 1)]
                if head_id == 0:
                    head_token_ids = []
                else:
                    head_token_ids = token_range[subtoken_map == (head_id - 1)]

                dep_rel_label = torch.tensor([self.dep_tag2id.get(deprel, 0)])
                for token_id in token_ids:
                    for head_token_id in head_token_ids:
                        graph.add_edges(token_id, head_token_id, data={'dep_link': dep_rel_label,
                                                                       'dtype': torch.tensor([0])})

                # self loop
                dep_rel_label = torch.tensor([self.dep_tag2id['cyclic']])
                for token_id1 in token_ids:
                    for token_id2 in token_ids:
                        graph.add_edges(token_id1, token_id2, data={'dep_link': dep_rel_label,
                                                                    'dtype': torch.tensor([0])})

                # link roots between two adjacent sentences
                dep_rel_label = torch.tensor([self.dep_tag2id['<pad>']])
                if root_token_ids is not None and head_id == 0:
                    for root_token_id in root_token_ids:
                        for token_id in token_ids:
                            graph.add_edges(token_id, root_token_id, data={'dep_link': dep_rel_label,
                                                                           'dtype': torch.tensor([0])})
                            graph.add_edges(root_token_id, token_id, data={'dep_link': dep_rel_label,
                                                                           'dtype': torch.tensor([0])})
                if head_id == 0:
                    root_token_ids = token_ids

        # predicate and argument nodes
        # argument -> predicate
        # word -> argument & predicate
        predicates = srl_dict.keys()
        num_predicates = len(predicates)
        graph.add_nodes(num_predicates)
        node_id_offset = num_tokens
        graph.ndata['unit'][node_id_offset:] = torch.ones(num_predicates) * 1
        graph.ndata['dtype'][node_id_offset:] = torch.ones(num_predicates) * 1
        predicate2nid = {predicate: i + node_id_offset for i, predicate in enumerate(predicates)}
        arguments = set()
        for _, args in srl_dict.items():
            for arg_start, arg_end, _ in args:
                arguments.add((arg_start, arg_end))
        num_arguments = len(arguments)
        graph.add_nodes(num_arguments)
        node_id_offset += num_predicates
        graph.ndata['unit'][node_id_offset:] = torch.ones(num_arguments) * 2
        graph.ndata['dtype'][node_id_offset:] = torch.ones(num_arguments) * 2
        arg2nid = {(arg_start, arg_end): i + node_id_offset for i, (arg_start, arg_end) in enumerate(arguments)}
        for predicate in predicates:
            predicatenid = predicate2nid[predicate]
            for arg_start, arg_end, label in srl_dict[predicate]:
                argnid = arg2nid[(arg_start, arg_end)]
                graph.add_edges(argnid, predicatenid, data={'srl_link': torch.tensor([label]),
                                                            'dtype': torch.tensor([1])})
                graph.add_edges(predicatenid, argnid, data={'srl_link': torch.tensor([label]),
                                                            'dtype': torch.tensor([1])})

        for arg_start, arg_end in arguments:
            argnid = arg2nid[(arg_start, arg_end)]
            for i in range(arg_start, arg_end + 1):
                graph.add_edges(i, argnid, data={'ta_link': torch.randint(10, (1,)),
                                                 'dtype': torch.tensor([2])})
                graph.add_edges(argnid, i, data={'ta_link': torch.randint(10, (1,)),
                                                 'dtype': torch.tensor([2])})

        arg_ids = []
        arg_mask = []
        for arg_start, arg_end in arguments:
            arg_ids.append(list(range(arg_start, arg_end + 1)))
        if len(arg_ids) != 0:
            max_arg_len = max([len(arg) for arg in arg_ids])
            for i in range(len(arg_ids)):
                arg_len = len(arg_ids[i])
                if arg_len < max_arg_len:
                    arg_ids[i] += [0] * (max_arg_len - arg_len)
                arg_mask.append([1] * arg_len + [0] * (max_arg_len - arg_len))

        arg_ids = torch.tensor(arg_ids, dtype=torch.int64)
        arg_mask = torch.tensor(arg_mask, dtype=torch.int64)
        predicates = torch.tensor(list(predicates), dtype=torch.int64)

        return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
                sentence_map, subtoken_map, graph, arg_ids, arg_mask, predicates)


class CoNLLDataLoader(object):
    def __init__(self, config, tokenizer, mode="train"):
        if mode == "train":
            self.train_batch_size = 1
            self.eval_batch_size = 1
            self.test_batch_size = 1
        else:
            self.test_batch_size = 1

        self.config = config
        self.tokenizer = tokenizer
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.dep_tag2id = util.get_dep_tag_vocab(config['dep_vocab_file'])
        adjunct_roles, core_roles = util.split_srl_labels()
        srl_label_dict_inv = [""] + adjunct_roles + core_roles
        self.srl_tag2id = {label: i for i, label in enumerate(srl_label_dict_inv)}

    def convert_examples_to_features(self, data_path):
        with open(data_path) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]

        data_instances = []
        for example in examples:
            data_instances.append(tensorize_example(example, self.config, self.tokenizer, self.genres, self.srl_tag2id))

        return data_instances

    def get_dataloader(self, data_sign="train"):
        if data_sign == 'train':
            features = self.convert_examples_to_features(self.config['train_path'])
            dependencies = util.read_depparse_features(self.config['train_dep_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, dependencies, self.dep_tag2id, sign='train')
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        elif data_sign == 'eval':
            features = self.convert_examples_to_features(self.config['eval_path'])
            dependencies = util.read_depparse_features(self.config['eval_dep_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, dependencies, self.dep_tag2id, sign='eval')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.eval_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        else:
            features = self.convert_examples_to_features(self.config['test_path'])
            dependencies = util.read_depparse_features(self.config['test_dep_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, dependencies, self.dep_tag2id, sign='test')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size, num_workers=16,
                                    collate_fn=collate_fn)

        return dataloader


def tensorize_example(example: dict, config: dict, tokenizer: BertTokenizer, genres: dict,
                      srl_tag2id: dict) -> CoNLLCorefResolution:
    clusters = example["clusters"]
    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = [0] * len(gold_mentions)
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    cluster_ids = torch.tensor(cluster_ids, dtype=torch.int64)

    sentences = example["sentences"]
    num_words = sum(len(s) + 2 for s in sentences)
    speakers = example["speakers"]
    speaker_dict = util.get_speaker_dict(util.flatten(speakers), config['max_num_speakers'])

    max_sentence_length = config['max_segment_len']
    text_len = torch.tensor([len(s) for s in sentences], dtype=torch.int64)

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
        sentence = ['[CLS]'] + sentence + ['[SEP]']
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [-1] + [1] * (len(sent_input_ids) - 2) + [-1]
        sent_speaker_ids = [1] + [speaker_dict.get(s, 3) for s in speaker] + [1]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(0)
            sent_input_mask.append(0)
            sent_speaker_ids.append(0)
        input_ids.append(sent_input_ids)
        speaker_ids.append(sent_speaker_ids)
        input_mask.append(sent_input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_mask = torch.tensor(input_mask, dtype=torch.int64)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
    assert num_words == torch.sum(torch.abs(input_mask)), (num_words, torch.sum(torch.abs(input_mask)))

    doc_key = example["doc_key"]
    subtoken_map = torch.tensor(example.get("subtoken_map", None), dtype=torch.int64)
    sentence_map = torch.tensor(example['sentence_map'], dtype=torch.int64)
    genre = genres.get(doc_key[:2], 0)
    genre = torch.tensor([genre], dtype=torch.int64)
    gold_starts, gold_ends = tensorize_mentions(gold_mentions)
    srl_dict = util.read_srl_features(example['srl'], srl_tag2id)

    return CoNLLCorefResolution(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                cluster_ids, sentence_map, subtoken_map, srl_dict)


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []

    starts = torch.tensor(starts, dtype=torch.int64)
    ends = torch.tensor(ends, dtype=torch.int64)
    return starts, ends


def truncate_example(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, srl_dict, config):
    max_training_sentences = config["max_training_sentences"]
    num_sentences = input_ids.size()[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    subtoken_map = subtoken_map[word_offset: word_offset + num_words]
    gold_spans = (gold_ends >= word_offset) & (gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    srls = {}
    for predicate, arguments in srl_dict.items():
        if word_offset <= predicate < word_offset + num_words:
            predicate -= word_offset
            srls[predicate] = []
            for arg_start, arg_end, label in arguments:
                arg_start -= word_offset
                arg_end -= word_offset
                srls[predicate].append((arg_start, arg_end, label))

    return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map, srls)


def collate_fn(data):
    data = zip(*data)
    data = [x[0] for x in data]
    return data

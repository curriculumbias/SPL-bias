from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

class ElectraDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 32,
        doc_max_len: int = 256,
        max_input: int = 1280000,
        task: str = 'ranking',
        num_buckets: int = 1000,
        sigma: float = 1.0
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = query_max_len + doc_max_len + 3
        self._max_input = max_input
        self._task = task
        self._num_buckets = num_buckets
        self._sigma = sigma

        if self._seq_max_len > 512:
            raise ValueError('query_max_len + doc_max_len + 3 > 512.')

        # Load data
        self._load_data()
        self._count = len(self._examples)
        self._buckets = self._create_buckets()

    def _load_data(self):
        # Similar data loading logic as in BertDataset, adapted as necessary
        # depending on `dataset` format and content type

        # Populate self._examples based on `self._dataset`
        if isinstance(self._dataset, str):
            self._id = False
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    if self._mode != 'train' or self._dataset.split('.')[-1] == 'json' or self._dataset.split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        if self._task == 'ranking':
                            query, doc_pos, doc_neg = line.strip('\n').split('\t')
                            line = {'query': query, 'doc_pos': doc_pos, 'doc_neg': doc_neg}
                        elif self._task == 'classification':
                            query, doc, label = line.strip('\n').split('\t')
                            line = {'query': query, 'doc': doc, 'label': int(label)}
                        else:
                            raise ValueError('Task must be `ranking` or `classification`.')
                    self._examples.append(line)
        elif isinstance(self._dataset, dict):
            self._id = True
            self._queries = {}
            with open(self._dataset['queries'], 'r') as f:
                for line in f:
                    if self._dataset['queries'].split('.')[-1] == 'json' or self._dataset['queries'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        query_id, query = line.strip('\n').split('\t')
                        line = {'query_id': query_id, 'query': query}
                    self._queries[line['query_id']] = line['query']
            self._docs = {}
            with open(self._dataset['docs'], 'r') as f:
                for line in f:
                    if self._dataset['docs'].split('.')[-1] == 'json' or self._dataset['docs'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        doc_id, doc = line.strip('\n').split('\t')
                        line = {'doc_id': doc_id, 'doc': doc}
                    self._docs[line['doc_id']] = line['doc']
            if self._mode == 'dev':
                qrels = {}
                with open(self._dataset['qrels'], 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] not in qrels:
                            qrels[line[0]] = {}
                        qrels[line[0]][line[2]] = int(line[3])
            with open(self._dataset['trec'], 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = line.strip().split()
                    if self._mode == 'dev':
                        if line[0] not in qrels or line[2] not in qrels[line[0]]:
                            label = 0
                        else:
                            label = qrels[line[0]][line[2]]
                    if self._mode == 'train':
                        if self._task == 'ranking':
                            self._examples.append({'query_id': line[0], 'doc_pos_id': line[1], 'doc_neg_id': line[2]})
                        elif self._task == 'classification':
                            self._examples.append({'query_id': line[0], 'doc_id': line[1], 'label': int(line[2])})
                        else:
                            raise ValueError('Task must be `ranking` or `classification`.')
                    elif self._mode == 'dev':
                        self._examples.append({'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    elif self._mode == 'test':
                        self._examples.append({'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    else:
                        raise ValueError('Mode must be `train`, `dev` or `test`.')
        else:
            raise ValueError('Dataset must be `str` or `dict`.')
        pass  # Implement similar to BertDataset

    ########################################################
    def _create_buckets(self):
        buckets = [[] for _ in range(self._num_buckets)]
        for i, example in enumerate(self._examples):
            bucket_index = i % self._num_buckets
            buckets[bucket_index].append(example)
        return buckets

    def _create_weights(self):
        weights = []
        num_buckets = len(self._buckets)
        
        # Create a Gaussian distribution centered at the first bucket
        mu = 0  # Mean (center of the Gaussian) at the first bucket
        sigma = self._sigma  # Standard deviation (adjust this to control the spread)
        
        for i, bucket in enumerate(self._buckets):
            # Calculate the Gaussian weight for this bucket
            bucket_weight = np.exp(-0.5 * ((i - mu) / sigma)**2)
            print(bucket_weight)
            # All items within the bucket have equal weight
            bucket_weights = [bucket_weight] * len(bucket)
            weights.extend(bucket_weights)
        
        # Normalize weights so they sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return normalized_weights

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'ranking':
                input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                segment_ids_pos = torch.tensor([item['segment_ids_pos'] for item in batch])
                input_mask_pos = torch.tensor([item['input_mask_pos'] for item in batch])
                input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                segment_ids_neg = torch.tensor([item['segment_ids_neg'] for item in batch])
                input_mask_neg = torch.tensor([item['input_mask_neg'] for item in batch])
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg}
            elif self._task == 'classification':
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['segment_ids'] for item in batch])
                input_mask = torch.tensor([item['input_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': label}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        elif self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')


    def pack_electra_features(self, query_tokens: List[str], doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
        input_mask = [1] * len(input_tokens)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        return input_ids, input_mask, segment_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._id:
            example['query'] = self._queries[example['query_id']]
            if self._mode == 'train' and self._task == 'ranking':
                example['doc_pos'] = self._docs[example['doc_pos_id']]
                example['doc_neg'] = self._docs[example['doc_neg_id']]
            else:
                example['doc'] = self._docs[example['doc_id']]
        if self._mode == 'train':
            if self._task == 'ranking':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens_pos = self._tokenizer.tokenize(example['doc_pos'])[:self._seq_max_len-len(query_tokens)-3]
                doc_tokens_neg = self._tokenizer.tokenize(example['doc_neg'])[:self._seq_max_len-len(query_tokens)-3]

                input_ids_pos, input_mask_pos, segment_ids_pos = self.pack_electra_features(query_tokens, doc_tokens_pos)
                input_ids_neg, input_mask_neg, segment_ids_neg = self.pack_electra_features(query_tokens, doc_tokens_neg)
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg}
            elif self._task == 'classification':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]
                
                input_ids, input_mask, segment_ids = self.pack_electra_features(query_tokens, doc_tokens)
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': example['label']}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]

            input_ids, input_mask, segment_ids = self.pack_electra_features(query_tokens, doc_tokens)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        elif self._mode == 'test':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]

            input_ids, input_mask, segment_ids = self.pack_electra_features(query_tokens, doc_tokens)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')


    def __len__(self) -> int:
        return self._count
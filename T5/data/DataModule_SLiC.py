from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import torch

from transformers import T5TokenizerFast

from data.ProcessData import KorCorpus
import jsonlines
import json
import random


class SourceRetrievalDataModule(LightningDataModule):
    def __init__(self, _config, dist=True):
        super().__init__()
        self._config = _config
        self.per_gpu_batch_size = _config['per_gpu_batch_size']
        self.num_workers = _config['num_workers']
        self.input_seq_len = _config['input_seq_len']
        self.dist = dist
        
        self.train_dataset_path = _config['train_dataset_path']
        self.val_dataset_path = _config['val_dataset_path']
        self.test_dataset_path = _config['test_dataset_path']
        self.model_name = _config['model_name']
        
    def setup(self, stage=None):
        if self._config['mode'] == 'test':
            self.test_dataset = LoadDataset(self._config, self.model_name, self.test_dataset_path, seq_len=self.input_seq_len, mode=self._config['mode'])
            
            # self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_dataset = LoadDataset(self._config, self._config["train_dataset_path"])  
            self.val_dataset = LoadDataset(self._config, self._config["val_dataset_path"])
            
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        
    def traindataset_reinit_at_train_epoch_end(self, current_epoch_plus):
        self.train_dataset = LoadDataset(self._config) 
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_gpu_batch_size,
            # sampler=self.test_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    
    
class LoadDataset(Dataset):
    def __init__(self, _config, data_path):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.encoder_seq_len = _config["input_seq_len"]
        self.decoder_seq_len = _config["label_seq_len"]
        corpus_path = data_path
        self.tokenizer = T5TokenizerFast.from_pretrained(_config["model_name"])

        if _config["mode"] == "train":
            data = KorCorpus.load(corpus_path)
            self._data = self.make_dataset(data, _config)
        else:
            data = KorCorpus.load(corpus_path)
            self._data = self.make_dataset_test(data, _config)

# for SLiC
    def make_dataset(self, data, _config):
        # input_ids
        # positive
        # negative
        # labels
        processed_data = []
        json_list = []
        
        read_file_name = "candidate.jsonl"
        with jsonlines.open(read_file_name) as f:
            for line in f:
                json_output = line["output"]
                json_label = line["label"]
                score = line["ROUGE-1"]
                
                json_list.append({"candidate" : json_output, "label" : json_label, "score" : score})
        
        for i in range(len(data)):
            sentence1 = data[i]['sentence1']
            sentence3 = data[i]['sentence3']
            output = data[i]['output']
            inst = "[문장1]과 [문장3] 사이에 들어갈 맥락에 맞는 [문장2]를 생성하세요.\n"
            
            source = inst + "[문장1] " + sentence1 + "\n" + "[문장2] " + "<extra_id_0> " + "\n" + "[문장3] " + sentence3
            
            source_ids = self.tokenizer.encode_plus(source,
                                              add_special_tokens=True,
                                              return_token_type_ids=False,
                                              )
            
            target = "[문장2] " + "<extra_id_0> " + output
            
            target_ids = self.tokenizer.encode_plus(target,
                                              add_special_tokens=True,
                                              return_token_type_ids=False,
                                              )
            
            input_form = source_ids.input_ids
            
            if self.encoder_seq_len >= len(input_form):
                input_padding = (self.encoder_seq_len - len(input_form)) * [self.tokenizer.pad_token_id]
                input_ids = input_form + input_padding
                input_attn_mask = source_ids.attention_mask + (len(input_padding) * [0])
            else:
                # truncation
                input_ids = input_form[: self.encoder_seq_len]
                input_attn_mask = [1] * self.encoder_seq_len
            
            assert len(input_ids) == len(input_attn_mask)
            
            output_form = target_ids.input_ids
            
            if self.decoder_seq_len >= len(output_form):
                output_padding = (self.decoder_seq_len - len(output_form)) * [self.tokenizer.pad_token_id]
                output_ids = output_form + output_padding
                output_attn_mask = target_ids.attention_mask + (len(output_padding) * [0])
            else:
                # truncation
                output_ids = output_form[: self.decoder_seq_len]
                output_attn_mask = [1] * self.decoder_seq_len
            
            assert len(output_ids) == len(output_attn_mask)
            
            # positive candidate
            if json_list[i]["score"] > 0.52:
                positive_candidate = "[문장2] <extra_id_0> " + json_list[i]["candidate"]
                negative_candidate = "[문장2] <extra_id_0> " + json_list[random.randint(0, 15000)]["candidate"]
            else:
                positive_candidate = "[문장2] <extra_id_0> " + json_list[i]["label"]
                negative_candidate = "[문장2] <extra_id_0> " + json_list[i]["candidate"]
                
            positive_candidate = self.tokenizer.encode_plus(positive_candidate, add_special_tokens=True, return_token_type_ids=False).input_ids
            negative_candidate = self.tokenizer.encode_plus(negative_candidate, add_special_tokens=True, return_token_type_ids=False).input_ids
            
            if self.encoder_seq_len >= len(positive_candidate):
                positive_candidate_padding = (self.encoder_seq_len - len(positive_candidate)) * [self.tokenizer.pad_token_id]
                positive_candidate = positive_candidate + positive_candidate_padding
            else:
                positive_candidate = positive_candidate[: self.encoder_seq_len]
            
            if self.encoder_seq_len >= len(negative_candidate):
                negative_candidate_padding = (self.encoder_seq_len - len(negative_candidate)) * [self.tokenizer.pad_token_id]
                negative_candidate = negative_candidate + negative_candidate_padding
            else:
                negative_candidate = negative_candidate[: self.encoder_seq_len]
            
            processed_data.append({"input_ids":input_ids,
                                    "attention_mask":input_attn_mask,
                                    "positive_candidate":positive_candidate,
                                    "negative_candidate":negative_candidate,
                                    "labels": output_ids})
            
        return processed_data
    
    def make_dataset_test(self, data, _config):
        data = list(data)
        processed_data = []
        
        for i in range(len(data)):
            title = data[i]['title']
    
            text = self.tokenizer.encode_plus(title, add_special_tokens=True, padding='max_length', max_length= _config['input_seq_len'], return_attention_mask=True)
            
            processed_data.append({"input_ids":text['input_ids'],
                                    "attention_mask":text['attention_mask']})
            
        return processed_data

    def __getitem__(self, index) -> dict:
        input = self._data[index]
        
        input_dict = {}
        input_dict.update(input)
        
        
        return {k:torch.tensor(v) for k, v in input_dict.items()}
    
    def __len__(self):
        return len(self._data)
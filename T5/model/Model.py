from typing import (
    Optional,
    Tuple,
    Union
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_config,
    get_peft_model
    )
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    )
import pytorch_lightning as pl
import os
import bitsandbytes
from rouge_metric import PyRouge
from SLiC import SLiC
from model.ft_model import FTModule
from deepspeed.ops.adam import FusedAdam

            
class SourceRetrievalModule(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters()
        self._config = _config
        self.training_loss = None
        self.val_losses = []
        self.ROUGEs = []
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if self._config["quantization"]:
    
            
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            )
            
            self.model = T5ForConditionalGeneration.from_pretrained(self._config["model_name"],
                                                                torch_dtype="auto",
                                                                quantization_config=bnb_config,
                                                                trust_remote_code=True,
                                                                device_map=local_rank,
                                                                )

        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self._config["model_name"],
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code=True,
                                                                device_map=local_rank,
                                                                )
            # for inference
            # self.model = T5ForConditionalGeneration(AutoConfig.from_pretrained(self._config["model_name"]))
            # self.model = self.model.to(torch.bfloat16)
        if self._config["peft"]:
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                            inference_mode=False,
                            r=self._config["lora_rank"],
                            lora_alpha=16,
                            lora_dropout=0.05,
                            bias=self._config["lora_train_bias"],
                            target_modules=_config["target_modules"],
                            )
            self.model = get_peft_model(self.model, lora_config)
        
        # for gradient checkpointing
        if self._config["gradient_checkpointing"]:
            self.model.base_model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.base_model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
        print(f"Gradient checkpointing: {self.model.base_model.is_gradient_checkpointing}")
    
        self.tokenizer = T5TokenizerFast.from_pretrained(_config['model_name'])
        # for SLiC
        # ckpt_path = ".ckpt"
        # self.__ft_model = FTModule.load_from_checkpoint(ckpt_path, _config=self._config)
        # self.__ft_model.cuda()
                
        
    def training_step(self, batch, batch_idx):
        output = self.model(**batch, use_cache=True)
        
        self.log(f"train/loss", output.loss, prog_bar=True)
        
        return output.loss
    
    # for SLiC
    # def training_step(self, batch, batch_idx):
    #     # ckpt_path = ".ckpt"
    #     # ft_model = FTModule.load_from_checkpoint(ckpt_path, _config=self._config)
    #     # ft_model = ft_model.to(torch.bfloat16)
    #     # ft_model.cuda()
    #     # ft_model.freeze()
        
    #     slic = SLiC()
        
    #     prompt, prompt_attn, positive_candidate, negative_candidate, labels = batch["input_ids"], batch["attention_mask"], batch["positive_candidate"], batch["negative_candidate"], batch["labels"]
    #     loss = slic(self.model, prompt, prompt_attn, positive_candidate, negative_candidate, labels)
    #     mean_loss = F.relu(loss.mean())
    #     mean_loss.requires_grad_()
        
    #     self.log(f"train/Closs", mean_loss, prog_bar=True)
        
    #     return mean_loss
    
    # def validation_step(self, batch, batch_idx):
    #     output = self.model(**batch, use_cache=True)
    #     # for gradient_checkpointing
    #     # output.loss.requires_grad_(True)
        
    #     self.val_losses.append(output.loss)
    #     self.log(f"val/loss", output.loss, prog_bar=True)
        
    #     return output.loss

    def ROUGE(self, output, label):
        candidate = [output]
        references = [
            [label]
        ]
        rouge = PyRouge(rouge_n=(1, 2, 4))
        rouge_score = rouge.evaluate(list(map(lambda cdt: " ".join(cdt), candidate)), \
                                    list(map(lambda refs: [" ".join(ref) for ref in refs], references)))
        
        return rouge_score["rouge-1"]["f"]
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        
        generated = self.tokenizer.batch_decode(self.model.generate(input_ids=input_ids, attention_mask=attention_mask), skip_special_tokens=True)
        generated = generated
        labels = self.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        labels = labels
        
        """
        generated, labels 형식
        -> [문장2]  생성 문장
        
        + generated는 처음에는 [문장2]  를 생성x
        """
        
        score = self.ROUGE(generated, labels)
        
        self.ROUGEs.append(score)
        
        self.log(f"val/ROUGE", score, prog_bar=True)
        
        return score
    
    def on_validation_epoch_end(self):
        mean_ROUGE = sum(self.ROUGEs) / len(self.ROUGEs)
        
        self.log(f"val/mean_ROUGE", mean_ROUGE, prog_bar=True)
    
        self.ROUGEs.clear()     
        
    def configure_optimizers(self):
        param_optimizer = self.named_parameters()
        no_decay = ['bias']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            self._config['lr']
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams._config['lr'], betas=self._config["betas"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams._config['warmup_steps'], num_training_steps=self.hparams._config['max_steps']
        )

        sched = {"scheduler": scheduler, "interval": "step"}
        

        return (
            [optimizer],
            [sched],
        )
    
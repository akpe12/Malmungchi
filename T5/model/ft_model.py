import pytorch_lightning as pl
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    )

class FTModule(pl.LightningModule):
    def __init__(self, _config) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self._config = _config
        self.model_name = "paust/pko-t5-large"
        self.model = T5ForConditionalGeneration(AutoConfig.from_pretrained(self.model_name))
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        
        
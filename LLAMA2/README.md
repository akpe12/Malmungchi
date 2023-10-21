# malmungchi
```
train
cli command : deepspeed --include localhost:3 --master_port 25999 full_finetun_lora --train_mode='lora'
inference
CUDA_VISIBLE_DEVICES=2 python full_finetun_generate.py
```
## Hyper parameters
```python
# - train
lr = 1e-5,
epoch = 1,
save_steps = 300,
seed = 42,

# - lora
lora_alpha = 32,
r = 8,
lora_dropout = 0.05,

# - tokenizer 
max_length = 200 
padding_side = 'left'

# - model
batch_size = 40
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# - Optimizer/scheduler 
# you can check this parameters in ds_config.json

# - Quantization config
load_in_4bit=True
bnb_4bit_compute_dtype=torch.bfloat16
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
```
## Train/Inference-prompt
```python
# - Train
'''
32토큰 이하로 두 문장 사이에 올 자연스러운 문장을 작성하세요 -\n\n
순서: 문장1, 문장2, 문장3 \n\n
문장1 : {data_point["input"]["sentence1"]} \n\n
문장3 : {data_point["input"]["sentence3"]} \n\n
문장2 : {data_point["output"]}
'''

# - Inference
'''
문장1 : 우리는 의자를 큰 책상 주위에 빙 둘러놓았다.
문장3 : 학생들이 모두 착석한 뒤 회의가 시작되었다.
문장2 : 그러자 의자에 학생들이 하나 둘 앉기 시작했다.

32토큰 이하로 두 문장 사이에 올 자연스러운 문장을 작성하세요 -\n\n
순서: 문장1, 문장2, 문장3 \n\n
문장1 : {data_point["input"]["sentence1"]} \n\n
문장3 : {data_point["input"]["sentence3"]} \n\n
문장2 :
'''
```
## Inference parameters
batch = 4
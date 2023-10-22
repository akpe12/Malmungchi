# T5 with SLiC

## Comparison
![점수](https://github.com/akpe12/Malmungchi/assets/77143331/3b6829cc-bf10-454c-848d-4007c885783f)

## Prompt
- T5는 pretraining에서 span corruption이 적용된 mlm 학습을 적용했던 모델. Mask가 존재하는 곳에 어떤 token들이 존재했는지 알아맞히는 이 pretraining의 objective는 현재 우리가 해결해야 하는 task의 objective와 같다고 생각하여 pretraining에서와 같이 input 형태를 만들어주면 될 것 같다고 판단.

![image](https://github.com/akpe12/Malmungchi/assets/77143331/a76800b0-0e1e-4247-8de0-1e8ae30a8aad)

그리하여, 세 모델에 동일하게 적용된 input 그리고 label의 형태는 아래와 같음.

```python
# - Train
'''
[문장1]과 [문장3] 사이에 들어갈 맥락에 맞는 [문장2]를 생성하세요.
[문장1] sentence1
[문장2] <extra_id_0>
[문장3] sentence3
'''

# - Inference
'''
[문장1]과 [문장3] 사이에 들어갈 맥락에 맞는 [문장2]를 생성하세요.
[문장1] sentence1
[문장2] <extra_id_0>
[문장3] sentence3
'''
```

## Hyper parameters
```python
# - train
lr = 2e-4,
epoch = 4 ~ 5,
seed = 42,

# - model
batch_size = 256
per_gpu_batch_size = 32

# you can check more details below
# config
# ㄴ config.py
```

## Inference parameters
```python 
single batch inference

# - generation configuration
max length = 250
num_beams = 5
```

## Training environment
```python
1 NVIDIA TITAN RTX
```

## Training times
```python
Fine-tuning T5: 15 hrs
Including further training for SLiC: 32 hrs
```

# SLiC

## Paper
CALIBRATING SEQUENCE LIKELIHOOD IMPROVES CONDITIONAL LANGUAGE GENERATION(Zhao et al., 2022)

![image](https://github.com/akpe12/Malmungchi/assets/77143331/7e6d0f1f-3137-489c-ba21-10e213a24fa6)

- 기존 Fine-tuning은 MLE만 가지고 진행되므로 모델이 생성해내는 문장의 품질은 낮을 가능성이 존재.
- Fine-tuning 모델이 생성한 candidates와 SLiC training을 위한 새로운 loss function을 정의하여  further training을 진행함으로써 모델이 calibrated된 문장을 생성할 수 있도록 함.
- 이 방법론을 통해 모델은 더 품질이 좋은 문장을 생성할 수 있게 될 것이라고 기대했으며, 점수 향상에 도움을 줄 것이라고 판단하여 적용.

## SLiC summary
![image](https://github.com/akpe12/Malmungchi/assets/77143331/be5d2e3f-3dd9-448a-985a-a7dbf1b78354)

## Similarity function
- SLiC training 시 사용될 positive candidate와 negative candidate를 구분하기 위한 similarity function으로는 ROUGE score를 채택하였음.

## Objective function
Rank loss  
![image](https://github.com/akpe12/Malmungchi/assets/77143331/d1a70fb7-7e49-4b7f-9444-505ac128a152)

Cross-entropy loss (as regularizer)   
![image](https://github.com/akpe12/Malmungchi/assets/77143331/52896a1f-81a9-4548-9fc2-7b703a1e1f75)

## SLiC training graph
![image](https://github.com/akpe12/Malmungchi/assets/77143331/8fa03068-e523-46e5-91f2-d04421cc5512)
![image](https://github.com/akpe12/Malmungchi/assets/77143331/a7c3d64b-7949-4c3e-877e-0c199daa566f)

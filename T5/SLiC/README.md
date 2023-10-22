# SLiC

## Paper
CALIBRATING SEQUENCE LIKELIHOOD IMPROVES CONDITIONAL LANGUAGE GENERATION(Zhao et al., 2022)

사진

- 기존 Fine-tuning은 MLE만 가지고 진행되므로 모델이 생성해내는 문장의 품질은 낮을 가능성이 존재.
- Fine-tuning 모델이 생성한 candidates와 SLiC training을 위한 새로운 loss function을 정의하여  further training을 진행함으로써 모델이 calibrated된 문장을 생성할 수 있도록 함.
- 이 방법론을 통해 모델은 더 품질이 좋은 문장을 생성할 수 있게 될 것이라고 기대했으며, 점수 향상에 도움을 줄 것이라고 판단하여 적용.

## SLiC summary
사진

## Similarity function
- SLiC training 시 사용될 positive candidate와 negative candidate를 구분하기 위한 similarity function으로는 ROUGE score를 채택하였음.

## Objective function
Rank loss
사진

Cross-entropy loss (as regularizer)
사진

## SLiC training graph
사진

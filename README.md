# Parameter-Efficient Multilingual Adaptation: GPT-2 & Qwen2-1.5B on XNLI

> SUTD 교환학기 NLP Final Project (팀 4인) — 본인 담당: Task 2 (English NLI Fine-tuning)



## 1. 프로젝트 개요

Full fine-tuning은 성능은 좋지만 모델 크기가 커질수록 비용이 크고 재사용성이 낮다는 문제가 있습니다.
이 프로젝트는 GPT-2를 직접 구현하는 것부터 시작해, English NLI fine-tuning, 멀티링구얼 zero-shot/fine-tuning 실험,
그리고 LoRA 기반 파라미터 효율적 적응까지 단계적으로 확장한 팀 프로젝트입니다.

| 단계 | 내용 | 핵심 결과 |
|---|---|---|
| Task 1 | GPT-2 아키텍처 + AdamW 옵티마이저 직접 구현 | HuggingFace 구현체와 수치 일치 검증 |
| **Task 2** | **English NLI Fine-tuning** | **81.50% (LR sweep 통한 최적화)** |
| Task 3 | 15개 언어 zero-shot / per-language / unified 멀티링구얼 실험 | tokenizer fertility ↔ accuracy 음의 상관관계 규명 |
| Extension | LoRA vs Full FT, GPT-2 vs Qwen2-1.5B 비교 | Qwen2 LoRA로 크로스링구얼 성능 대폭 개선  |



## 2. Task 2: English NLI Fine-tuning

#### 문제 설정
NLI(Natural Language Inference)를 GPT-2의 next-token generation 문제로 프레이밍:

```
Premise: <premise_text>
Hypothesis: <hypothesis_text>
Label: (entailment / neutral / contradictory)
```

모델이 Label 토큰을 생성하도록 supervised fine-tuning을 수행하고, prefix 부분은 loss에서 마스킹(`-100`) 처리.

#### 구현 내용
- GPT-2 forward pass를 통해 얻은 hidden state를 vocabulary logit으로 변환 (`hidden_state_to_token`, weight-tying 방식)
- Next-token prediction loss 계산 (logits/labels shift + `CrossEntropyLoss(ignore_index=-100)`)
- Greedy decoding 기반 텍스트 생성 함수 구현 (EOS 조기 종료 포함)
- AdamW 옵티마이저 자체 구현 및 참조 구현체 대비 수치 검증

#### Learning Rate Sweep 실험

| Config | Learning Rate | Avg Loss | Test Accuracy |
|---|---|---|---|
| 1 (Baseline) | 5e-5 | 0.6139 | 78.58% |
| 2 | 5e-4 | 1.0005 | 52.24% |
| 3 | 1e-6 | 0.7662 | 76.33% |
| **4 (Best)** | **2e-5** | **0.5820** | **81.50%** |

- LR이 너무 크면(5e-4) 학습이 발산에 가깝게 불안정해지고 정확도 급락
- LR이 너무 작으면(1e-6) 수렴이 느려 성능 손실
- 최적 LR(2e-5)에서 baseline 대비 +2.92%p 개선

#### 결론
GPT-2는 1 epoch의 supervised fine-tuning만으로도 English NLI에서 강한 성능(81%+)에 도달하며,
이는 이후 Task 3(멀티링구얼)과 Extension(LoRA) 실험의 baseline이 됨.



## 3. 프로젝트 전체 파이프라인

#### Task 3: 멀티링구얼 실험
- 15개 언어에 대해 tokenizer fertility(단어당 평균 subword 수) 측정 → accuracy와 음의 상관관계 확인
- Fertility < 2.5인 언어(en, de, es, fr, sw) 선정하여 per-language / unified fine-tuning 비교

| Language | Zero-Shot | Per-Lang | Unified |
|---|---|---|---|
| English | 78.34% | 78.34% | 72.30% |
| Spanish | 46.87% | 66.49% | 60.58% |
| French | 41.74% | 63.93% | 59.42% |
| German | 40.28% | 65.39% | 59.34% |
| Swahili | 38.54% | 57.96% | 53.77% |

#### Extension: LoRA 기반 파라미터 효율적 적응
- GPT-2 LoRA: 전체 파라미터의 0.94%만 업데이트하고도 Full FT와 동등한 성능(79.20% vs 78.58%)
- Qwen2-1.5B LoRA: 멀티링구얼 프리트레이닝 덕분에 English-only 학습만으로도 강한 zero-shot 크로스링구얼 전이 (다수 언어 70~90%)
- 결론: 크로스링구얼 성능을 좌우하는 건 fine-tuning 전략보다 베이스 모델의 멀티링구얼 프리트레이닝 여부



## 4. 기술 스택
`PyTorch` `HuggingFace Transformers` `PyTorch Lightning` `AdamW (custom impl.)` `Google Colab (T4 GPU)`



## 5. Report
"Parameter-Efficient Multilingual Adaptation: A Comparative Study of GPT-2 and Qwen2–1.5B on XNLI" (팀 공동 작성, 본인은 Task 2 섹션 및 관련 실험 작성)

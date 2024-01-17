# ERA-SESSION27

🤗[**Space Link**]



### Tasks:
1. :heavy_check_mark: Use OpenAssistant dataset.
2. :heavy_check_mark: Finetune Microsoft Phi2 model.
3. :heavy_check_mark: Use QLoRA stratergy.
4. :heavy_check_mark: Create an App on HF space using finetuned model.

## Phi2 Model Description:
```python
PhiForCausalLM(
  (transformer): PhiModel(
    (embd): Embedding(
      (wte): Embedding(51200, 2560)
      (drop): Dropout(p=0.0, inplace=False)
    )
    (h): ModuleList(
      (0-31): 32 x ParallelBlock(
        (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
        (mixer): MHA(
          (rotary_emb): RotaryEmbedding()
          (Wqkv): Linear4bit(in_features=2560, out_features=7680, bias=True)
          (out_proj): Linear4bit(in_features=2560, out_features=2560, bias=True)
          (inner_attn): SelfAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
          (inner_cross_attn): CrossAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (mlp): MLP(
          (fc1): Linear4bit(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear4bit(in_features=10240, out_features=2560, bias=True)
          (act): NewGELUActivation()
        )
      )
    )
  )
  (lm_head): CausalLMHead(
    (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=2560, out_features=51200, bias=True)
  )
  (loss): CausalLMLoss(
    (loss_fct): CrossEntropyLoss()
  )
)
```

### Training Output
```python
TrainOutput(global_step=500, training_loss=1.711106170654297, metrics={'train_runtime': 5222.3118, 'train_samples_per_second': 1.532, 'train_steps_per_second': 0.096, 'total_flos': 3.293667738832896e+16, 'train_loss': 1.711106170654297, 'epoch': 0.81})
```
### Loss vs Steps Logs
![image](https://github.com/GunaKoppula/ERAV1-Session-27/assets/61241928/585180c8-c299-43d9-a9ac-dbead74555ca)

## Sample Results:
![image](https://github.com/GunaKoppula/ERAV1-Session-27/assets/61241928/e448436a-3bb4-4d48-b6d1-8cbfd72cae13)

![image](https://github.com/GunaKoppula/ERAV1-Session-27/assets/61241928/008fad10-4f7b-4df2-8e16-361cf4ae4776)

## Gradio UI:

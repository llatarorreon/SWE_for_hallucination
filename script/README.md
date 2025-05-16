### Environment Setup
Before running, set your OpenAI API base and key:

```bash
export OPENAI_API_BASE=https://your-api-endpoint/v1
export OPENAI_API_KEY=your-api-key
```

### Inference

Run model inference on a specific risk setting using the following command (scenario and result directory are fixed):

```bash
python inference_swebench.py --risk-setting <risk-setting-name> --scenario swebench --result-dir ../results/swebench --models <model-name>
```

---

### Verifier

After inference, evaluate the model's outputs using the risk-aware verifier:

```bash
python verifier.py --type <risk-setting-name> --scenario swebench --inference-results-dir ../results/swebench --model <model-name>
```
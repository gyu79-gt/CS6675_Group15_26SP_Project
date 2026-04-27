# CS6675 Final Project Submission

This folder contains the final dataset, reranking pipelines, shared evaluator, live demo script, and saved results for our project on reranking web evidence for binary political forecasting.

The Group Memembers are: Guangyi Yu, Jingfei Zhou, Christina Zhang, Yuxuan Liang, Yuchen Han

## Structure

- `data/search_data.json`: final cleaned evidence dataset used by all rerankers
- `data/scripts/`: scripts used for data collection, filtering, labeling, and leak checking
- `pipelines/`: reranking scripts from team members
- `eval/eval.py`: shared downstream LLM evaluation script
- `demo/demo.py`: live command line demo using Exa search and local LLM reranking
- `ranking_results/`: saved reranked outputs and evaluation summaries

## Dataset

Final dataset: `data/search_data.json`

- 348 binary political prediction market questions
- 20 cleaned web search documents per question
- 145 YES / 203 NO resolved labels
- evidence restricted to documents before each question cutoff date
- shared input used by all submitted reranking pipelines


## API Keys and Local Models

Scripts that use Exa or OpenAI expect a local `api_keys.json` with key names like:

```json
{
  "EXA": "...",
  "OPENAI": "..."
}
```

Local LLM scripts expect an LM Studio compatible server at:

```bash
http://127.0.0.1:1234/v1
```

Our final local runs used Qwen 3.5 27B with local quantized models.

## Run Evaluation

Example evaluation with local Qwen 3.5 27B 4 bit quantized:

```bash
python3 eval/eval.py --input ranking_results/reranked_stance_jing.jsonl --provider lmstudio --model qwen3.5-27b@q4_k_xl --output ranking_results/acc_results/results_reranked_stance_jing_top10.jsonl
```

Example OpenAI evaluation:

```bash
python3 eval/eval.py --input ranking_results/search_data_hybridbm25_leon.json --provider openai --model gpt-5.4-mini --workers 4 --output ranking_results/acc_results_openai/results_search_data_hybridbm25_leon_top10_openai.jsonl
```

## Run Demo

```bash
python3 demo/demo.py --question "Will Brent Crude Oil close above $120 on April 15th, 2026?" --description "This market resolves YES if Brent Crude Oil closes at or above $120 per barrel on April 15, 2026."
```

## Results

Saved reranked outputs are in `ranking_results/`.

Saved evaluation outputs are in:

- `ranking_results/acc_results/`: local Qwen evaluator
- `ranking_results/acc_results_openai/`: OpenAI evaluator

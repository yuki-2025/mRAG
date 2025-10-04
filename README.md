# mRAG : Multimodal RAG — Paper Q&amp;A System Based on Qwen2VL mini-4o + Evaluation

This repo introduces the full process of building an mRAG (Multimodal Retrieval-Augmented Generation) application and provides detailed explanations of its key principles.
 
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/13d6c33d-6341-4a29-996c-115a4d5c1c93" />

Please refer to my [blog](https://yuki-blog1.vercel.app/article/mRAG) for the code explanation and complete details.


## Python Environment

- Install environment

```bash
pip install -r requirements.txt
```


## Download Data

> So all the required data is already listed in this script.

```bash
bash download_data.sh
```

## Downlaod Model

> If you need to download them in parts, please refer to the script comments.

```bash
bash download_models.sh
```

## SFT

- single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python qwen25vl_sft.py
```

- multi GPU

```bash
accelerate launch --config_file accelerate_config.yaml qwen25vl_sft.py

# For background execution, it’s best to change to absolute paths.
nohup accelerate launch --config_file accelerate_config.yaml qwen25vl_sft.py > logs/output_pt.log 2>&1 &
```

## mRAG

- Data Synthesis 

```bash
CUDA_VISIBLE_DEVICES=0 python mini_vlm/qwen25vl_mRAG_eval_data.py
```

- Evaluate
> Need DeepSeek API Key

```bash
CUDA_VISIBLE_DEVICES=0 python mini_vlm/qwen25vl_mRAG_eval.py
```

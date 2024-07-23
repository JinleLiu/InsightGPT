# InsightGPT (Pioneering Urban Traffic Scene Insight Ability Through Enhanced Model Training Strategy)

<p align="center">
    <br>
    <img src="Resources/Visualized.png"/>
    <br>
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.5-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
</p>


## 📖 Table of Contents
- [Introduction](#-introduction)
- [Installation](#%EF%B8%8F-installation)
- [Getting Started](#-getting-started)
- [Classroom](#-Classroom)
- [License](#-License)
- [Citation](#-citation)

## 📝 Introduction
The InsightGPT model significantly enhances the understanding capabilities of multimodal generative AI in intelligent transportation systems through innovative data organization and training strategies. By combining human annotators with a powerful general-purpose Multimodal Large Language Model (MLLM), we optimized the organization of data to ensure efficient and high-quality information processing. Additionally, InsightGPT employs a phased training strategy that inputs data for different training objectives into specific learning rate phases, guided by a Warmup-Stable-Decay learning rate scheduler. This method not only deepens the model's understanding of the overall traffic scene but also enhances its ability to recognize and reason about individual objects within the scene.

SWIFT web-ui is available both on [Huggingface space](https://huggingface.co/spaces/tastelikefeet/swift) and [ModelScope studio](https://www.modelscope.cn/studios/iic/Scalable-lightWeight-Infrastructure-for-Fine-Tuning/summary), please feel free to try!



## 🛠️ Installation

SWIFT runs in the Python environment. Please ensure your Python version is higher than 3.8.

- Method 1: Install SWIFT using pip command:

```shell
# Full capabilities
pip install 'ms-swift[all]' -U
# LLM only
pip install 'ms-swift[llm]' -U
# AIGC only
pip install 'ms-swift[aigc]' -U
# Adapters only
pip install ms-swift -U
```

- Method 2: Install SWIFT through source code (convenient for running training and inference scripts), please run the following commands:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

SWIFT depends on torch>=1.13, recommend torch>=2.0.0.

- Method 3: Use SWIFT in our Docker image

```shell
# China-Hangzhou image
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
# US-west image
docker pull registry.us-west-1.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
```

## 🚀 Getting Started

This section introduces basic usage, see the [Documentation](https://swift.readthedocs.io/en/latest/) section for more ways to use.

### Web-UI

Web-UI is a gradio-based interface for **zero-threshold** training and deployment. It is easy to use and perfectly supports multi-GPU training and deployment:

```shell
SWIFT_UI_LANG=en swift web-ui
```

![image.png](./docs/resources/web-ui-en.jpg)

### Training

#### Training Scripts
You can refer to the following scripts to customize your own training script.

- full: [qwen1half-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat/full) (A100), [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_mp) (2\*A100)
- full+ddp+zero2: [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_ddp_zero2) (4\*A100)
- full+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3) (4\*A100)
- lora: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora) (3090), [baichuan2-13b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/baichuan2_13b_chat/lora_mp) (2\*3090), [yi-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat/lora) (A100), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_mp) (2\*A100)
- lora+ddp: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora_ddp) (2\*3090)
- lora+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/lora_ddp_zero3) (4\*3090), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_ddp_zero3) (4\*A100)
- qlora(gptq-int4): [qwen-7b-chat-int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int4/qlora) (3090)
- qlora(gptq-int8): [qwen1half-7b-chat-int8](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_int8/qlora) (3090)
- qlora(bnb-int4): [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/qlora) (3090)


#### Supported Training Processes

| Training Process | Training Method                                                               |
|------------------|-------------------------------------------------------------------------------|
| Pretraining      | Text Generation                                                               |
| Fine-tuning      | Single-turn/Multi-turn<br>Agent Training/Self-cognition<br>Multi-modal Vision/Multi-modal Speech|
| Human Alignment  | DPO<br>ORPO<br>SimPO                                                          |
| Text-to-Image    | DreamBooth, etc.                                                              |
| Text-to-Video    | -                                                                             |

#### Single GPU Training

Start single GPU fine-tuning with the following command:

LoRA:
```shell
# Experimental Environment: A100
# GPU Memory Requirement: 20GB
# Runtime: 3.1 hours
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --eval_steps 200 \
```

Full-parameter:
```shell
# Experimental Environment: A100
# GPU Memory Requirement: 80GB
# Runtime: 2.5 hours
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type full \
    --output_dir output \
    --eval_steps 500 \
```


#### Model Parallel Training


```shell
# Experimental Environment: 2 * A100
# GPU Memory Requirement: 10GB + 13GB
# Runtime: 3.4 hours
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```

#### Data Parallel Training

```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 30GB
# Runtime: 0.8 hours
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```

Combining Model Parallelism and Data Parallelism:
```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 2*14GB + 2*18GB
# Runtime: 1.7 hours
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```

#### Deepspeed Training
Deepspeed supports training of quantized GPTQ and AWQ models.

ZeRO2:
```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 21GB
# Runtime: 0.9 hours
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --deepspeed default-zero2 \
```

ZeRO3:
```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 19GB
# Runtime: 3.2 hours
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --deepspeed default-zero3 \
```

ZeRO3-Offload:
```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 12GB
# Runtime: 60 hours
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_id_or_path AI-ModelScope/WizardLM-2-8x22B \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --deepspeed zero3-offload \
```


#### Multi-node Multi-GPU
```shell
# If the disk is not shared, please additionally specify `--save_on_each_node true` in the shell scripts on each machine.
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=8 \
swift sft \
    --model_type qwen1half-32b-chat \
    --sft_type full \
    --dataset blossom-math-zh \
    --output_dir output \
    --deepspeed default-zero3 \

# node1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=8 \
swift sft \
    --model_type qwen1half-32b-chat \
    --sft_type full \
    --dataset blossom-math-zh \
    --output_dir output \
    --deepspeed default-zero3 \
```

##### AliYun-DLC multi-node training
In DLC product, WORLD_SIZE is the node number, RANK is the node index, this is different from the definition of torchrun.

```shell
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
swift sft \
    --model_type qwen1half-32b-chat \
    --sft_type full \
    --dataset blossom-math-zh \
    --output_dir output \
    --deepspeed default-zero3
```

#### Pretraining

```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 30GB
# Runtime: 0.8 hours
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model_type qwen1half-7b \
    --dataset chinese_c4#10000 \
    --num_train_epochs 1 \
    --sft_type full \
    --deepspeed default-zero3 \
    --output_dir output \
```


#### RLHF

```shell
# We support rlhf_type dpo/cpo/simpo/orpo/kto
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type dpo \
    --model_type qwen1half-7b-chat \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```


### Inference
Original model:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
# use VLLM
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat \
    --infer_backend vllm --max_model_len 8192
```

LoRA fine-tuned:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir xxx/checkpoint-xxx --load_dataset_config true
# use VLLM
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir xxx/checkpoint-xxx --load_dataset_config true \
    --merge_lora true --infer_backend vllm --max_model_len 8192
```

### Evaluation

Original model:
```shell
# We recommend using vLLM for acceleration (arc evaluated in half a minute)
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen1half-7b-chat \
    --eval_dataset ARC_e --infer_backend vllm
```

LoRA fine-tuned:
```shell
CUDA_VISIBLE_DEVICES=0 swift eval --ckpt_dir xxx/checkpoint-xxx \
    --eval_dataset ARC_e --infer_backend vllm \
    --merge_lora true \
```

### Quantization

Original model:
```shell
CUDA_VISIBLE_DEVICES=0 swift export --model_type qwen1half-7b-chat \
    --quant_bits 4 --quant_method awq
```

LoRA fine-tuned:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir xxx/checkpoint-xxx --load_dataset_config true \
    --quant_method awq --quant_bits 4 \
    --merge_lora true \
```

### Deployment
The client uses the OpenAI API for invocation, for details refer to the [LLM deployment documentation](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/VLLM-inference-acceleration-and-deployment.md).

Original model:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen1half-7b-chat
# 使用VLLM加速
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen1half-7b-chat \
    --infer_backend vllm --max_model_len 8192
```

LoRA fine-tuned:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir xxx/checkpoint-xxx
# 使用VLLM加速
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --ckpt_dir xxx/checkpoint-xxx --merge_lora true \
    --infer_backend vllm --max_model_len 8192
```

### Supported Models
The complete list of supported models and datasets can be found at [Supported Models and Datasets List](docs/source_en/LLM/Supported-models-datasets.md).

#### LLMs

| Model Type                                                                                      | Model Introduction                                                                                                                             | Language           | Model Size                                | Model Type                                                        |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------------------------------|-------------------------------------------------------------------|
| Qwen<br>Qwen1.5<br>Qwen2                                                                        | [Tongyi Qwen 1.0 and 1.5 series models](https://github.com/QwenLM)                                                                             | Chinese<br>English | 0.5B-110B<br>including quantized versions | base model<br>chat model<br>MoE model<br>code model               |
| ChatGLM2<br>ChatGLM3<br>Codegeex2<br>GLM4<br>Codegeex4                                          | [Zhipu ChatGLM series models](https://github.com/THUDM)                                                                                        | Chinese<br>English | 6B-9B                                     | base model<br>chat model<br>code model<br>long text model         |
| Baichuan<br>Baichuan2                                                                           | [Baichuan 1 and Baichuan 2](https://github.com/baichuan-inc)                                                                                   | Chinese<br>English | 7B-13B<br>including quantized versions    | base model<br>chat model                                          |
| Yuan2                                                                                           | [Langchao Yuan series models](https://github.com/IEIT-Yuan)                                                                                    | Chinese<br>English | 2B-102B                                   | instruct model                                                    |
| XVerse                                                                                          | [XVerse series models](https://github.com/xverse-ai)                                                                                           | Chinese<br>English | 7B-65B                                    | base model<br>chat model<br>long text model<br>MoE model          |
| LLaMA2                                                                                          | [LLaMA2 series models](https://github.com/facebookresearch/llama)                                                                              | English            | 7B-70B<br>including quantized versions    | base model<br>chat model                                          |
| LLaMA3                                                                                          | [LLaMA3 series models](https://github.com/meta-llama/llama3)                                                                                   | English            | 8B-70B<br>including quantized versions    | base model<br>chat model                                          |
| Mistral<br>Mixtral                                                                              | [Mistral series models](https://github.com/mistralai/mistral-src)                                                                              | English            | 7B-22B                                    | base model<br>instruct model<br>MoE model                         |
| Yi<br>Yi1.5                                                                                     | [01AI's YI series models](https://github.com/01-ai)                                                                                            | Chinese<br>English | 6B-34B<br>including quantized             | base model<br>chat model<br>long text model                       |
| InternLM<br>InternLM2<br>InternLM2-Math<br>InternLM2.5                                          | [Pujiang AI Lab InternLM series models](https://github.com/InternLM/InternLM)                                                                  | Chinese<br>English | 1.8B-20B                                  | base model<br>chat model<br>math model                            |
| DeepSeek<br>DeepSeek-MoE<br>DeepSeek-Coder<br>DeepSeek-Math<br>DeepSeek-V2<br>DeepSeek-Coder-V2 | [DeepSeek series models](https://github.com/deepseek-ai)                                                                                       | Chinese<br>English | 1.3B-236B                                 | base model<br>chat model<br>MoE model<br>code model<br>math model |
| MAMBA                                                                                           | [MAMBA temporal convolution model](https://github.com/state-spaces/mamba)                                                                      | English            | 130M-2.8B                                 | base model                                                        |
| Gemma<br>Gemma2                                                                                 | [Google Gemma series models](https://github.com/google/gemma_pytorch)                                                                          | English            | 2B-27B                                    | base model<br>instruct model                                      |
| MiniCPM                                                                                         | [OpenBmB MiniCPM series models](https://github.com/OpenBMB/MiniCPM)                                                                            | Chinese<br>English | 2B-3B                                     | chat model<br>MoE model                                           |
| OpenBuddy                                                                                       | [OpenBuddy series models](https://github.com/OpenBuddy/OpenBuddy)                                                                              | Chinese<br>English | 7B-70B                                    | base model<br>chat model                                          |
| Orion                                                                                           | [OrionStar AI series models](https://github.com/OrionStarAI)                                                                                   | Chinese<br>English | 14B                                       | base model<br>chat model                                          |
| BlueLM                                                                                          | [VIVO BlueLM large model](https://github.com/vivo-ai-lab/BlueLM)                                                                               | Chinese<br>English | 7B                                        | base model<br>chat model                                          |
| Ziya2                                                                                           | [Fengshenbang series models](https://github.com/IDEA-CCNL/Fengshenbang-LM)                                                                     | Chinese<br>English | 13B                                       | base model<br>chat model                                          |
| Skywork                                                                                         | [Skywork series models](https://github.com/SkyworkAI/Skywork)                                                                                  | Chinese<br>English | 13B                                       | base model<br>chat model                                          |
| Zephyr                                                                                          | Zephyr series models based on Mistral                                                                                                          | English            | 7B                                        | chat model                                                        |
| PolyLM                                                                                          | [Tongyi Lab self-developed PolyLM series models](https://github.com/DAMO-NLP-MT/PolyLM)                                                        | Multilingual       | 13B                                       | base model                                                        |
| SeqGPT                                                                                          | [Tongyi Lab self-developed text understanding model for information extraction and text classification](https://github.com/Alibaba-NLP/SeqGPT) | Chinese            | 560M                                      | semantic understanding model                                      |
| SUS                                                                                             | [Southern University of Science and Technology model fine-tuned on YI](https://github.com/SUSTech-IDEA/SUS-Chat)                               | Chinese<br>English | 34B                                       | chat model                                                        |
| Tongyi-Finance                                                                                  | [Tongyi finance series models](https://github.com/QwenLM/Qwen)                                                                                 | Chinese<br>English | 14B                                       | base model<br>chat model<br>financial model                       |
| CodeFuse-CodeLLaMA<br>CodeFuse-Codegeex2<br>CodeFuse-Qwen                                       | [Ant CodeFuse series models](https://github.com/codefuse-ai)                                                                                   | Chinese<br>English | 6B-34B                                    | chat model<br>code model                                          |
| phi2/phi3                                                                                       | Microsoft's PHI series models                                                                                                                  | English            | 3B/4B                                     | base model<br>instruct model<br>code model                        |
| Grok                                                                                            | [X-ai](https://github.com/xai-org/grok-1)                                                                                                      | English            | 300B                                      | base model                                                        |
| TeleChat                                                                                        | [Tele-AI](https://github.com/Tele-AI/Telechat)                                                                                                 | Chinese<br>English | 7B-12B                                    | chat model                                                        |
| dbrx                                                                                            | [databricks](https://github.com/databricks/dbrx)                                                                                               | English            | 132B                                      | base model<br>chat model                                          |
| mengzi3                                                                                         | [Langboat](https://github.com/Langboat/Mengzi3)                                                                                                | Chinese<br>English | 13B                                       | base model                                                        |
| c4ai-command-r                                                                                  | [c4ai](https://cohere.com/command)                                                                                                             | Multilingual       | 35B-104B                                  | chat model                                                        |
| WizardLM2                                                                                       | [WizardLM2 series models](https://github.com/nlpxucan/WizardLM)                                                                                | English            | 7B-8x22B<br>including quantized versions  | chat model<br>MoE model                                           |
| Atom                                                                                            | [Atom](https://github.com/LlamaFamily/Llama-Chinese)                                                                                           | Chinese            | 7B                                        | base model<br>chat model                                          |
| Chinese-LLaMA-Alpaca-2                                                                          | [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)                                                                      | Chinese            | 1.3B-13B                                  | base model<br>chat model<br>long text model                       |
| Chinese-LLaMA-Alpaca-3                                                                          | [Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)                                                                      | Chinese            | 8B                                        | base model<br>chat model                                          |
| ModelScope-Agent                                                                                | [ModelScope Agent series models](https://github.com/modelscope/modelscope-agent)                                                               | Chinese            | 7B-14B                                    | agent model                                                       |
| Numina                                                                                          | [AI-MO](https://huggingface.co/AI-MO)                                                                                                          | English            | 7B                                        | Math                                                              |

#### MLLMs

| Model Type                                              | Model Introduction                                                                     | Language           | Model Size                            | Model Type               |
|---------------------------------------------------------|----------------------------------------------------------------------------------------|--------------------|---------------------------------------|--------------------------|
| Qwen-VL                                                 | [Tongyi Qwen vision model](https://github.com/QwenLM)                                  | Chinese<br>English | 7B<br>including quantized versions    | base model<br>chat model |
| Qwen-Audio                                              | [Tongyi Qwen speech model](https://github.com/QwenLM)                                  | Chinese<br>English | 7B                                    | base model<br>chat model |
| YI-VL                                                   | [01AI's YI series vision models](https://github.com/01-ai)                             | Chinese<br>English | 6B-34B                                | chat model               |
| XComposer2<br>XComposer2.5                              | [Pujiang AI Lab InternLM vision model](https://github.com/InternLM/InternLM-XComposer) | Chinese<br>English | 7B                                    | chat model               |
| DeepSeek-VL                                             | [DeepSeek series vision models](https://github.com/deepseek-ai)                        | Chinese<br>English | 1.3B-7B                               | chat model               |
| MiniCPM-V<br>MiniCPM-V-2<br>MiniCPM-V-2_5               | [OpenBmB MiniCPM vision model](https://github.com/OpenBMB/MiniCPM)                     | Chinese<br>English | 3B-9B                                 | chat model               |
| CogVLM<br>CogAgent<br>CogVLM2<br>CogVLM2-Video<br>GLM4V | [Zhipu ChatGLM visual QA and Agent model](https://github.com/THUDM/)                   | Chinese<br>English | 9B-19B                                | chat model               |
| Llava1.5<br>Llava1.6                                    | [Llava series models](https://github.com/haotian-liu/LLaVA)                            | English            | 7B-34B                                | chat model               |
| Llava-Next<br>Llava-Next-Video                          | [Llava-Next series models](https://github.com/LLaVA-VL/LLaVA-NeXT)                     | Chinese<br>English | 7B-110B                               | chat model               |
| mPLUG-Owl                                               | [mPLUG-Owl series models](https://github.com/X-PLUG/mPLUG-Owl)                         | English            | 11B                                   | chat model               |
| InternVL<br>Mini-InternVL<br>InternVL2                  | [InternVL](https://github.com/OpenGVLab/InternVL)                                      | Chinese<br>English | 1B-40B<br>including quantized version | chat model               |
| Llava-llama3                                            | [xtuner](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers)             | English            | 8B                                    | chat model               |
| Phi3-Vision                                             | Microsoft                                                                              | English            | 4B                                    | chat model               |
| PaliGemma                                               | Google                                                                                 | English            | 3B                                    | chat model               |
| Florence                                                | Microsoft                                                                              | English            | 0.23B-0.77B                           | chat model               |


#### Diffusion Models

| Model Type          | Model Introduction                                                    | Language | Model Type        |
|---------------------|----------------------------------------------------------------------|----------|------------------ |
| AnimateDiff         | [AnimateDiff animation model](https://github.com/guoyww/AnimateDiff) | English  | text-to-video     |
| SD1.5/SD2.0/SDXL    | [StabilityAI series diffusion models](https://github.com/Stability-AI) | English | text-to-image    |

### Supported Open Source Datasets

| Dataset Type        | Training Task   | Documentation                                                                                                                                                                                                                                |
|---------------------|:----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General             | Fine-tuning     | 🔥ruozhiba, 🔥ms-bench, 🔥alpaca-en(gpt4), 🔥alpaca-zh(gpt4), multi-alpaca, instinwild, cot-en, cot-zh, firefly-zh, instruct-en, gpt4all-en, sharegpt, tulu-v2-sft-mixture, wikipedia-zh, open-orca, sharegpt-gpt4, deepctrl-sft, coig-cqia. |
| Agent               | Fine-tuning     | 🔥ms-agent, 🔥ms-agent-for-agentfabric, ms-agent-multirole, 🔥toolbench-for-alpha-umi, damo-agent-zh, damo-agent-zh-mini, agent-instruct-all-en.                                                                                             |
| General             | Human Alignment | hh-rlhf, 🔥hh-rlhf-cn, stack-exchange-paired.                                                                                                                                                                                                |
| Code                | Fine-tuning     | code-alpaca-en, 🔥leetcode-python-en, 🔥codefuse-python-en, 🔥codefuse-evol-instruction-zh.                                                                                                                                                  |
| Medical             | Fine-tuning     | medical-en, medical-zh, 🔥disc-med-sft-zh.                                                                                                                                                                                                   |
| Legal               | Fine-tuning     | lawyer-llama-zh, tigerbot-law-zh, 🔥disc-law-sft-zh.                                                                                                                                                                                         |
| Math                | Fine-tuning     | 🔥blossom-math-zh, school-math-zh, open-platypus-en.                                                                                                                                                                                         |
| SQL                 | Fine-tuning     | text2sql-en, 🔥sql-create-context-en.                                                                                                                                                                                                        |
| Text Generation     | Fine-tuning     | 🔥advertise-gen-zh, 🔥dureader-robust-zh.                                                                                                                                                                                                    |
| Classification      | Fine-tuning     | cmnli-zh, 🔥jd-sentiment-zh, 🔥hc3-zh, 🔥hc3-en.                                                                                                                                                                                             |
| Quantization Assist | Quantization    | pileval.                                                                                                                                                                                                                                     |
| Other               | Fine-tuning     | finance-en, poetry-zh, webnovel-zh, generated-chat-zh, cls-fudan-news-zh, ner-jave-zh.                                                                                                                                                       |
| Vision              | Fine-tuning     | coco-en, 🔥coco-en-mini, coco-en-2, coco-en-2-mini, capcha-images.                                                                                                                                                                           |
| Audio               | Fine-tuning     | aishell1-zh, 🔥aishell1-zh-mini.                                                                                                                                                                                                             |

### Supported Technologies

| Technology Name                                                                                                                                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🔥LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)                                                                                          |
| 🔥LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf)                                                                                   |
| 🔥GaLore:[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)                                                                      |
| 🔥LISA: [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919)                                                   |
| 🔥UnSloth: https://github.com/unslothai/unsloth                                                                                                                                         |
| 🔥LLaMA PRO: [LLAMA PRO: Progressive LLaMA with Block Expansion](https://arxiv.org/pdf/2401.02415.pdf)                                                                                  |
| 🔥SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  \ |
| 🔥NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)                                                                                          |
| LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)                                                                               |
| Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)                                                                                               |
| Vision Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)                                                                                                          |
| Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)                                                                     |
| Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  \  |
| Tuners provided by [PEFT](https://github.com/huggingface/peft), such as IA3, AdaLoRA, etc.                                                                                              |

### Supported Hardware

| Hardware Environment           | Notes                                           |
|--------------------------------|-------------------------------------------------|
| CPU                            |                                                 |
| RTX 20/30/40 series, etc.      | After 30 series, BF16 and FlashAttn can be used |
| Computing cards T4/V100, etc.  | BF16 and FlashAttn not supported                |
| Computing cards A10/A100, etc. | Support BF16 and FlashAttn                      |
| Huawei Ascend NPU              |                                                 |

### Environment variables

- DATASET_ENABLE_CACHE: Enable cache when preprocess dataset, you can use `1/True` or `0/False`, default `False`
- WEBUI_SHARE: Share your web-ui, you can use `1/True` or `0/False`, default `False`
- SWIFT_UI_LANG: web-ui language, you can use `en` or `zh`, default `zh`
- WEBUI_SERVER: web-ui host ip，`0.0.0.0` for all routes，`127.0.0.1` for local network only. Default `127.0.0.1`
- WEBUI_PORT: web-ui port
- USE_HF: Use huggingface endpoint or ModelScope endpoint to download models and datasets. you can use `1/True` or `0/False`, default `False`
- FORCE_REDOWNLOAD: Force to re-download the dataset

Other variables like `CUDA_VISIBLE_DEVICES` are also supported, which are not listed here.


## 📚 Classroom

| Tutorial Name                                                                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Introduction to Deep Learning](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/A.%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%E4%BB%8B%E7%BB%8D.md)                                                |
| [Large Model Basics](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/B.%E9%AD%94%E6%90%AD%E7%A4%BE%E5%8C%BA%E5%92%8CLLM%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86.md)                    |
| [Prompt Engineering](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/C.%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%B7%A5%E7%A8%8B-prompt%20engineering.md)                                                                 |
| [Transformer Architecture Introduction](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/D.Transformer%E7%BB%93%E6%9E%84.md)                                                                                   |
| [Training Technique Selection](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/E.%E6%8A%80%E6%9C%AF%E9%80%89%E5%9E%8B.md)                                                                                     |
| [Data Preprocessing](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/F.%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)                                                                                      |
| [Quantization](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/G.%E9%87%8F%E5%8C%96.md)                                                                                                                       |
| [Training](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/H.%E8%AE%AD%E7%BB%83.md)                                                                                                                           |
| [Inference](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/I.LLM%E5%92%8C%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E9%AB%98%E6%95%88%E6%8E%A8%E7%90%86%E5%AE%9E%E8%B7%B5.md)                             |
| [Deployment](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/J.%E9%83%A8%E7%BD%B2.md)                                                                                                                         |
| [Evaluation](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/K.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%87%AA%E5%8A%A8%E8%AF%84%E4%BC%B0%E7%90%86%E8%AE%BA%E5%92%8C%E5%AE%9E%E6%88%98--LLM%20Automatic%20Evaluation.md) |

## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## 📎 Citation

```bibtex
@Misc{swift,
  title = {SWIFT:Scalable lightWeight Infrastructure for Fine-Tuning},
  author = {The ModelScope Team},
  howpublished = {\url{https://github.com/modelscope/swift}},
  year = {2024}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)

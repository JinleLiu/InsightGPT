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

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
</p>


## 📖 Table of Contents
- [Introduction](#-introduction)
- [Installation](#%EF%B8%8F-installation)
- [Getting Started](#-getting-started)
- [License](#-License)
- [Citation](#-citation)

## 📝 Introduction
The InsightGPT model significantly enhances the understanding capabilities of multimodal generative AI in intelligent transportation systems through innovative data organization and training strategies. By combining human annotators with a powerful general-purpose Multimodal Large Language Model (MLLM), we optimized the organization of data to ensure efficient and high-quality information processing. Additionally, InsightGPT employs a phased training strategy that inputs data for different training objectives into specific learning rate phases, guided by a Warmup-Stable-Decay learning rate scheduler. This method not only deepens the model's understanding of the overall traffic scene but also enhances its ability to recognize and reason about individual objects within the scene.

The InsightGPT's parameters are open source on [Huggingface space](https://huggingface.co/JinLe/InsightGPT), please feel free to try!



## 🛠️ Installation

InsightGPT is adapted for the transportation domain based on GLM-4V-9b. The entire fine-tuning process is supported by the Swift platform. Therefore, before using InsightGPT, Swift must be installed first.You can seek the latest version installation in [ms-swift](https://github.com/modelscope/ms-swift), or download the "swift" folder from our repository and follow the steps below for installation.

SWIFT runs in the Python environment. Please ensure your Python version is higher than 3.8.

- Method 1: Install SWIFT using pip command:

```shell
# Full capabilities
pip install 'ms-swift[all]' -U
# LLM only
pip install 'ms-swift[llm]' -U
```

- Method 2: Install SWIFT through source code (convenient for running training and inference scripts), please run the following commands:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

SWIFT depends on torch>=1.13, recommend torch>=2.0.0.


## 🚀 Getting Started

This section introduces basic usage of InsightGPT, see the [Huggingface space](https://huggingface.co/JinLe/InsightGPT) for different versions download.
For example, use 'v0-20240709-182636-WSD10%'.(This is the best model version fine-tuned with 10% global step to annealing.)

- Method 1: Inference directly:

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir /root/autodl-tmp/ckpt/glm4v-9b-chat/v1-20240705-163125/checkpoint-2652 \
    --load_dataset_config true \
```
- Method 2: Inference after merge:

```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir /root/autodl-tmp/ckpt/glm4v-9b-chat/v1-20240705-163125/checkpoint-2652 \
    --merge_lora true
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

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)

# Evalverse
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/Evalverse_White.png" width=300>
  <source media="(prefers-color-scheme: light)" srcset="assets/Evalverse_Color.png" width=300>
  <img alt="Evalverse" src="assets/Evalverse_Color.png" width=300>
</picture>

The Universe of Evaluation.
All about the evaluation for LLMs.  </br>
Upstage Solar is powered by Evalverse! Try at Upstage [Console](https://console.upstage.ai/)!

[Docs](https://evalverse.gitbook.io/evalverse-docs) • [Examples](https://github.com/UpstageAI/evalverse/tree/main/examples) • [FAQ](https://evalverse.gitbook.io/evalverse-docs/documents/faqs) • [Contribution Guide](https://github.com/UpstageAI/evalverse/blob/main/contribution/CONTRIBUTING.md)  • [Contact](mailto:evalverse@upstage.ai)  • [Discord](https://discord.gg/D3bBj66K) • [Paper](https://arxiv.org/abs/2404.00943)
</div>

<div align="center"><img alt="overview" src="assets/overview.png" width=500></div>


## Welcome to Evalverse!
Evalverse is a freely accessible, open-source project designed to support your LLM (Large Language Model) evaluations. We provide a simple, standardized, and user-friendly solution for the processing and management of LLM evaluations, catering to the needs of AI research engineers and scientists. Even if you are not very familiar with LLMs, you can easily use Evalverse.
<div align="center"><img alt="architecture" src="assets/architecture.png" width=700></div>

</br>

## 🌌 Installation
### 🌠 Option 1: Git clone
Before cloning, please make sure you've set SSH key to your GitHub account.

#### 1. Clone the Evalverse repository
[Notes] `--recursive` for submodules
```
git clone --recursive https://github.com/UpstageAI/evalverse.git
```
#### 2. Install requirement packages
```
cd evalverse
pip install -e .
```

### 🌠 Option 2: Install via Pypi *(WIP)*
> **Currently, pip install is not supported. Please install Evalverse with option 1.**


</br>

## 🌌 Configuration
You have to set an API key or Token in the `.env` file (rename `.env_sample` to `.env`) to use some features.
- Your OpenAI API Key (required for `mt_bench`)
- Your Slack BOT/APP Token (required for slack reporter)
```
OPENAI_API_KEY=sk-...

SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

</br>

## 🌌 Quickstart
### 🌠 Evaluation with Library
The following code is a simple example to evaluate the [SOLAR-10.7B-Instruct-v1.0 model](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) from upstage on `h6_en`-[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

```python
import evalverse as ev

evaluator = ev.Evaluator()

model = "upstage/SOLAR-10.7B-Instruct-v1.0"
benchmark = "h6_en"

evaluator.run(model=model, benchmark=benchmark)
```


### 🌠 Evaluation with CLI
Here is a script that produces the same result as the above code:

```bash
cd evalverse

python3 evaluator.py \
--h6_en \
--ckpt_path upstage/SOLAR-10.7B-Instruct-v1.0
```

</br>

## 🌌 Contributors
<a href="https://github.com/UpstageAI/evalverse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=UpstageAI/evalverse" />
</a>


## 🌌 Acknowledgements
Evalverse is an open-source project orchestrated by the **Data-Centric LLM Team** at `Upstage`, designed as an ecosystem for LLM evaluation. Launched in April 2024, this initiative stands at the forefront of advancing evaluation handling in the realm of large language models (LLMs).

## 🌌 License
Evalverse is completely freely-accessible open-source and licensed under the Apache License 2.0.

## 🌌 Citation
If you want to cite our 🌌 Evalverse project, feel free to use the following bibtex. You can check our paper via [link](https://arxiv.org/abs/2404.00943).

```bibtex
@misc{evalverse,
  title = {Evalverse},
  author = {Jihoo Kim, Wonho Song, Dahyun Kim, Yoonsoo Kim, Yungi Kim, Chanjun Park},
  year = {2024},
  eprint={2404.00943},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

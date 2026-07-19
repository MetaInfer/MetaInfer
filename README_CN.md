<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/MetaInfer/MetaInfer/main/docs/logo.png" alt="MetaInfer" width="200" onerror="this.style.display='none'">
</p> -->

<h1 align="center">MetaInfer</h1>

<h3 align="center">
  <em>让AI自己生成 高性能、小巧、专一的大模型推理框架</em>
  <p>
  <em>大胆说出你的期望，MetaInfer帮你编写高性能专属推理框架</em>
</h3>

<h4 align="center">
  <em>（也是一个LLM 驱动的 AI Infra 优化工具包）</em>
</h3>

<p align="center">
  <a href="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml">
    <img src="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/metainfer">
    <img src="https://img.shields.io/pypi/v/metainfer?color=blue" alt="PyPI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-%3E%3D3.9-blue" alt="Python">
  </a>
  <a href="https://github.com/MetaInfer/MetaInfer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="https://www.star-history.com/?type=date&repos=MetaInfer%2FMetaInfer">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=MetaInfer/MetaInfer&type=date&theme=dark&legend=top-left&sealed_token=7N_57a34GhT7taYXyy9U_E1V_9P1i7A_0PK4Am3dOHxXcNvtk9CuxadGB6B1ZCyS0Zsa2rq_z1U0OmRgz9YDhWs5IaomukOlrF5zq5eapw47cM1rYdOKnQ" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=MetaInfer/MetaInfer&type=date&legend=top-left&sealed_token=7N_57a34GhT7taYXyy9U_E1V_9P1i7A_0PK4Am3dOHxXcNvtk9CuxadGB6B1ZCyS0Zsa2rq_z1U0OmRgz9YDhWs5IaomukOlrF5zq5eapw47cM1rYdOKnQ" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=MetaInfer/MetaInfer&type=date&legend=top-left&sealed_token=7N_57a34GhT7taYXyy9U_E1V_9P1i7A_0PK4Am3dOHxXcNvtk9CuxadGB6B1ZCyS0Zsa2rq_z1U0OmRgz9YDhWs5IaomukOlrF5zq5eapw47cM1rYdOKnQ" />
 </picture>
</a>
</p>

---

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 这是什么（我们的愿景）？

* ***让AI自己生成简单小巧，但性能持平甚至超越SOTA水平的推理框架。（杀手级应用）***

  * 现有的SOTA推理框架已经不堪重负
    * 传统软件工程中通过多层级抽象来适配多种模型、多种硬件、多种并行部署形式、多种量化方式、多种算子优化的方式已经不能满足当前大模型飞速发展的需要。
    * 推理框架中绝大多数代码在部署条件确定后是不必要的，然而这些兼容逻辑拖慢了系统性能。
  
  * 只要告诉MetaInfer你的期望，MetaInfer帮你编写满足你要求的高性能专属推理框架。
    * 模型类型是什么？在什么硬件上部署？并行方式是什么?使用了什么量化方式？优化吞吐还是优化延迟？

* ***让AI辅助优化算子***
  * 给定要优化的算子，大模型自动进行【优化-性能测试】组成的迭代循环

* ***模型迁移***
  * 新的模型在老旧GPU上还没有得到支持？给定高版本推理框架代码，由大模型帮助将其backport到低版本推理框架。

* ***嵌入式平台推理***
  * MetaInfer将C/C++、Rust也列为其支持目标。为未来边缘计算、具身智能场景提供支持。



### 内置任务类型

| 类型 | 说明 |
|---|---|
| **`gen-infer-framework`** | 构建模型专属推理服务器，提供 OpenAI 兼容 HTTP API。不可变 oracle 启动 `serve.sh`，发送固定 prompt，派发 LLM 裁判判定正确性。 |
| **`calc-theoretical-value`** | 计算 LLM 单次前向传播的理论 FLOPs 和显存带宽。完全只读的确定性流水线：模型检查 → 显存建模 → 计算图 → 可视化。 |
| **`example`** | 构建新任务类型的规范骨架。复制、重命名、取消 `register()` 注释、实现流水线——不改动任何共享代码。 |

## 快速开始

### 第一步，安装ccb。（本项目目前使开源的claude code版本，暂不支持其他coding agent，欢迎贡献代码以支持更多coding agent）

开源ccb项目地址：https://github.com/claude-code-best/claude-code
```
npm i -g claude-code-best
```

### 第二步，安装MetaInfer
```bash
git clone https://github.com/MetaInfer/MetaInfer.git
cd MetaInfer
pip install -r requirements.txt
./serve.py
```

打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，点击 **+ New Task**，
选择要做的任务，填写需求，大模型随即为您服务。

```bash
# 其他启动方式
./serve.py --host 0.0.0.0 --port 9000
METAINFER_PORT=9000 ./serve.py
python -m metainfer.server.app
```

## License

MIT

## 学术研究

MetaInfer的最初想法和实验数据已经公开在https://arxiv.org/abs/2607.12875，相关代码位于`arxiv-paper`分支。

引用信息：
```
@misc{miao2026metainferknowledgellminference,
      title={MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox}, 
      author={Zhenwen Miao and Honglin Wang and Mingheng Mi},
      year={2026},
      eprint={2607.12875},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2607.12875}, 
}
```

## 如何贡献
架构细节、设计理念和新任务类型添加方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。

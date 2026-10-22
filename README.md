# Qwen2.5 风格模型复现

这是一个面向学习和实验的 Qwen2.5-style decoder-only language model reproduction project。

当前目标是先复现公开的模型结构和训练链路，再根据服务器 GPU 资源扩大模型规模。项目不会把教学版 tokenizer、小规模数据或短训练结果称为官方 Qwen2.5 的完整复现；官方训练数据、清洗规则和全部训练细节并没有完全公开。

## 当前完成范围

- Qwen2.5-style 配置对象和 JSON 读取。
- RMSNorm。
- Rotary Position Embedding (RoPE)。
- Grouped Query Attention (GQA)。
- SwiGLU MLP。
- Causal decoder block。
- Causal language model head 和 next-token loss。
- 一个无外部依赖的字符级教学 tokenizer。
- 小数据集 packing、训练 batch 和单 batch 过拟合。
- 文档清洗、去重、文档级 train/valid 切分、DataLoader。
- 独立训练入口，支持 validation loss/PPL、梯度累积、BF16、warmup/cosine 和 resume。
- DDP 多卡训练 smoke 入口。
- 本地 Hugging Face config/tokenizer 适配入口，不会在 smoke test 中主动联网。
- KV cache 增量生成，以及 full/incremental logits 对齐测试。
- 每个模块都有独立的 `main_*.py` 验证入口。

## 目录

```text
qwen25_reproduction/
  qwen25/
    config.py       配置和参数检查
    tokenizer.py    教学 tokenizer 和官方 tokenizer 适配接口
    model.py        RMSNorm、RoPE、GQA、SwiGLU、decoder、LM head
    data.py         文档清洗、去重、切分、packing、DataLoader
    train.py        训练、评估和 checkpoint
  configs/
    tiny.json       CPU/单卡教学配置
    qwen25_0_5b_reference.json  0.5B 级参考配置，暂不直接训练
  main_config.py
  main_tokenizer.py
  main_components.py
  main_forward.py
  main_data.py
  main_data_pipeline.py
  main_overfit.py
  main_bf16.py
  main_metrics.py
  main_checkpoint.py
  main_ddp.py
  main_train.py
  main_train_pipeline.py
  main_official_adapter.py
  main_kv_cache.py
  tests/test_model.py
  notes/阶段1_模型结构与验证.md
  notes/阶段2_数据训练评测.md
  notes/阶段3_多卡训练.md
  notes/阶段4_官方资源接入.md
  notes/阶段5_KV缓存与生成.md
```

## 环境

推荐 Python 3.10+ 和 PyTorch 2.x。当前本机 Python 环境已检测到 PyTorch 2.10.0+cu128 和 1 张 CUDA GPU。

```powershell
cd C:\Users\lenovo\Documents\agent\qwen25_reproduction
python -m pip install torch
```

## 按模块运行

```powershell
python main_config.py
python main_tokenizer.py
python main_components.py
python main_forward.py
python main_data.py
python main_data_pipeline.py
python main_overfit.py
python main_bf16.py
python main_metrics.py
python main_checkpoint.py
python main_ddp.py
python main_train.py --steps 30
python main_train_pipeline.py --steps 30
python main_official_adapter.py --model-path C:\path\to\local\Qwen-model
python main_kv_cache.py
python -m unittest discover -s tests -v
```

在服务器上使用多张 GPU 时，运行：

```powershell
torchrun --standalone --nproc-per-node=GPU数量 main_ddp.py
```

如果直接运行 `main_*.py` 时 Python 找不到本地包，请确认当前目录是项目根目录。

## 推荐学习顺序

本机完整小规模验证入口：`python main_local_smoke.py --steps 100 --device cuda`。
运行结果写入 `artifacts/local_smoke/`，包括逐步指标、checkpoint 和 tokenizer。
实际结果与解读见 [阶段6：本机小规模测试](notes/阶段6_本机小规模测试.md)。

1. 先运行 `main_config.py`，理解配置如何决定张量形状。
2. 运行 `main_tokenizer.py`，观察文字如何变成 token id。
3. 运行 `main_components.py`，只看 RMSNorm、RoPE、GQA、SwiGLU 的输入输出。
4. 运行 `main_forward.py`，理解一个 batch 如何得到 logits 和 loss。
5. 运行 `main_overfit.py`，确认反向传播和参数更新工作。
6. 运行 `main_data_pipeline.py`，理解文档清洗、去重、切分、tokenize、pack 和 DataLoader。
7. 运行 `main_train_pipeline.py --steps 30`，观察 train/valid loss、PPL、学习率和 tokens/s。
8. 运行 `main_kv_cache.py`，理解推理时为什么缓存历史 K/V。
9. 运行 `main_official_adapter.py --model-path ...`，在服务器上接入本地官方 tokenizer/config。
10. 运行 `torchrun --standalone --nproc-per-node=GPU数量 main_ddp.py`，验证真正的多卡同步。
11. 按顺序阅读 `notes/阶段1_模型结构与验证.md` 到 `notes/阶段5_KV缓存与生成.md`。

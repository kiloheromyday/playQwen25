# 阶段 5：KV cache 和增量生成

## 1. 没有 KV cache 时为什么会重复计算

假设 prompt 有 100 个 token，模型已经生成了第 1 个新 token。生成第 2 个 token 时，如果把完整的 101 个 token 再送进模型，前面 100 个 token 的 query、key、value 都会重新计算。

但历史 token 的 key/value 并没有变化，所以可以缓存它们：

```text
第一次：完整 prompt -> 得到历史 K/V
后续：每次只输入 1 个新 token + 读取历史 K/V
```

这里的 `K` 是 key，`V` 是 value。Attention 仍然会让新 token 查询全部历史，但不再重新投影历史 token。

## 2. cache 的张量形状

本项目使用 GQA，所以缓存的是较少的 KV heads，而不是把 K/V 复制到所有 query heads 后再缓存：

```text
K/V cache: [batch, num_key_value_heads, cached_length, head_dim]
Q:         [batch, num_attention_heads, 1, head_dim]
```

计算 attention 前，K/V 才通过 `repeat_interleave` 扩展到 query head 数量。这样更节省显存，也符合 GQA 的设计目的。

## 3. 为什么要重新写 causal mask

全量计算时，第 `i` 个 query 只能看到位置 `0..i`。增量计算时，新 query 的绝对位置是 `past_length`，它可以看到 `0..past_length` 的所有 key。

因此 mask 不能永远使用一个方阵的上三角，而要比较：

```text
key_position > query_position
```

这也是缓存实现里最容易写错的地方之一。

## 4. 如何验证 cache 没有改变结果

`main_kv_cache.py` 做两个计算：

1. 一次性输入 `prompt + continuation`，得到 continuation 部分 logits；
2. 先输入 prompt，再每次输入一个 continuation token，同时传入上一轮的 K/V cache。

两者的最大 logits 差异应接近零。这个测试比只观察生成文本更可靠，因为文本可能因为 argmax 相同而掩盖数值错误。

运行：

```powershell
python main_kv_cache.py
```

## 5. 当前生成实现的边界

当前 `generate` 仍然是 greedy decoding，只取最大 logit；没有 temperature、top-k、top-p、beam search，也没有 paged attention。达到最大上下文长度后，它会重新截取窗口并重建 cache，保证位置长度不超过配置上限。

## 6. 向师兄汇报的说法

我给 decoder 加了 KV cache：首次前向保存每层历史 key/value，后续每一步只输入一个新 token；同时重写了带位置偏移的 causal mask。通过 full forward 和逐 token incremental forward 的 logits 对齐测试，验证缓存不会改变计算结果。当前生成是 greedy，后续如果做推理性能优化，再考虑采样策略和 paged KV cache。

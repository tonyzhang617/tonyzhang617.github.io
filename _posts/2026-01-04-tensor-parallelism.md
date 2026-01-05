---
layout: post
title: "The Elegance of Tensor Parallelism: Scaling LLMs Beyond a Single GPU"
date: 2026-01-04 22:54:33
description: An illustrative explanation of tensor parallelism for LLMs
tags: llms, parallelism, efficiency, gpus, ai-infra
categories: efficient-ai
markdown: true
---

Tensor parallelism is one of the most elegant algorithms for distributing large language models across multiple GPUs. Originally proposed in [Megatron-LM](https://arxiv.org/pdf/1909.08053), it has become essential for both training and inference of modern LLMs.

The core idea is simple: split each layer's weight matrices across available GPUs. During the forward pass, inputs are broadcast to every GPU, each GPU computes its portion of the layer, and outputs are combined using fast interconnects like NVLink. This process repeats layer by layer until the final output.

What makes tensor parallelism elegant is how it achieves efficiency through three design principles: *minimize communication*, *minimize redundant computation*, and *balance work evenly across GPUs*. The last principle is especially important—when GPUs finish at different times, the faster ones sit idle waiting for synchronization. By splitting work evenly, all GPUs stay fully utilized. This post explains how tensor parallelism achieves all three goals.


## Understanding LLM Architecture

Most modern LLMs are autoregressive transformer-based models composed of sequential transformer blocks. Each block in a Llama-style architecture contains four components: a normalization layer (RMSNorm), attention, an MLP, and residual connections. In PyTorch pseudocode:

```python
def transformer_block(x):
    x = x + attention(rms_norm(x))
    return x + mlp(rms_norm(x))
```

The attention mechanism projects the input into queries, keys, and values, computes attention scores, and projects the result back:

```python
def attention(x):
    q, k, v = W_q(x), W_k(x), W_v(x)
    scores = softmax((q @ k.T) / sqrt(d_head))
    return W_o(scores @ v)
```

The MLP consists of an up-projection, a gate, an activation function, and a down-projection:

```python
def mlp(x):
    return W_down(activation(W_up(x)) * W_gate(x))
```

Each of these weight matrices (`W_q`, `W_k`, `W_v`, `W_o`, `W_up`, `W_gate`, and `W_down`) can be parallelized across GPUs. The key question is *how* to split them.


## Column Parallel vs. Row Parallel

When distributing a weight matrix across N GPUs, we have two choices: split by columns or split by rows. These two strategies have fundamentally different implications for what data each GPU needs and what communication is required afterward.

**Column parallelism** splits the weight matrix along its output dimension, giving each GPU a vertical slice. Given an input `X`, each GPU computes `X @ Wᵢ` where `Wᵢ` is its slice. Since each GPU produces a different portion of the full result, the partial outputs `Y_i` must be concatenated (gathered) to reconstruct the complete output `Y`.

```markdown
        Column Parallel (split by output dimension)
        ════════════════════════════════════════════

            ┌────┬────┬────┬────┐              ┌────┬────┬────┬────┐
            │    │    │    │    │              │    │    │    │    │
        X @ │ W₀ │ W₁ │ W₂ │ W₃ │  ═══════▶    │ Y₀ │ Y₁ │ Y₂ │ Y₃ │
            │    │    │    │    │              │    │    │    │    │
            └────┴────┴────┴────┘              └────┴────┴────┴────┘
                                                        │
               (each GPU has full X)                    │
                                                        ▼
                                               ┌────────────────────────────────┐
                                               │  Y = Concatenate(Y₀,Y₁,Y₂,Y₃)  │
                                               └────────────────────────────────┘
```

**Row parallelism** partitions the weight matrix along its input dimension, assigning each GPU a horizontal slice. Since each row of the weight matrix corresponds to one input feature, row parallelism requires the input itself to be split by columns as well. Each GPU computes a local matrix product using its slice of `X` and its slice of `W`, producing a partial sum. These partial results `Y_i` must be summed (reduced) across GPUs to obtain the final output `Y`.

```markdown
        Row Parallel (split by input dimension)
        ════════════════════════════════════════

        X (split by columns)              W (split by rows)

        ┌────┬────┬────┬────┐            ┌──────────────────┐
        │    │    │    │    │            │        W₀        │
        │ X₀ │ X₁ │ X₂ │ X₃ │            ├──────────────────┤
        │    │    │    │    │            │        W₁        │
        └────┴────┴────┴────┘            ├──────────────────┤
                                         │        W₂        │
                                         ├──────────────────┤
                                         │        W₃        │
                                         └──────────────────┘

                         X₀ @ W₀  ═══▶  Y₀ ─┐
                         X₁ @ W₁  ═══▶  Y₁ ─┼──▶  Y = (Y₀ + Y₁ + Y₂ + Y₃)
                         X₂ @ W₂  ═══▶  Y₂ ─┤
                         X₃ @ W₃  ═══▶  Y₃ ─┘
```

The key insight is that **column parallel outputs concatenate, while row parallel outputs sum**. This distinction determines when each strategy is appropriate.


## Collective Communication Primitives
Before diving deeper, we need to understand two fundamental communication operations.

**All-Gather** collects data from all GPUs and distributes the complete result to everyone. If each GPU starts with a piece of data, after all-gather, every GPU has all the pieces concatenated together. This is useful when each GPU computed a different portion of an output and needs the full result.

**All-Reduce** sums data across all GPUs and distributes the sum to everyone. If each GPU has a partial result that needs to be combined additively, all-reduce performs the sum and ensures every GPU has the same final answer.

```markdown
        All-Gather
        ══════════
                                 
        GPU 0: [A]      ┐
        GPU 1: [B]      │ ═══▶  [A,B,C,D]
        GPU 2: [C]      │       on all
        GPU 3: [D]      ┘       GPUs
                                 
        (concatenation)


        All-Reduce
        ══════════
                                 
        GPU 0: [1,2]   ┐
        GPU 1: [3,4]   │ ═══▶  [10,14]
        GPU 2: [2,3]   │       on all
        GPU 3: [4,5]   ┘       GPUs
                                 
        (element-wise sum)
```

There's also **Reduce-Scatter**, which sums data across GPUs but gives each GPU only 1/N of the result. It's like all-reduce followed by splitting the output.

```markdown
        Reduce-Scatter
        ══════════════
                                 
        GPU 0: [1,2,3,4]   ┐        GPU 0: [4]
        GPU 1: [1,2,3,4]   │        GPU 1: [8]
        GPU 2: [1,2,3,4]   │ ═══▶   GPU 2: [12]
        GPU 3: [1,2,3,4]   ┘        GPU 3: [16]
                                 
        (element-wise sum, then scatter 1/N to each)
```

Importantly, **reduce-scatter + all-gather = all-reduce**. This equivalence becomes crucial for sequence parallelism.

```markdown
        Equivalence: Reduce-Scatter + All-Gather = All-Reduce
        ══════════════════════════════════════════════════════

        Starting state (same on all GPUs):
        ┌────────────────────────────────────────────────────────────────┐
        │  GPU 0: [1,2,3,4]                                              │
        │  GPU 1: [1,2,3,4]                                              │
        │  GPU 2: [1,2,3,4]                                              │
        │  GPU 3: [1,2,3,4]                                              │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │Reduce-Scatter│
                            └──────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  GPU 0: [4]      (sum of all 1st elements: 1+1+1+1)            │
        │  GPU 1: [8]      (sum of all 2nd elements: 2+2+2+2)            │
        │  GPU 2: [12]     (sum of all 3rd elements: 3+3+3+3)            │
        │  GPU 3: [16]     (sum of all 4th elements: 4+4+4+4)            │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │  All-Gather  │
                            └──────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  GPU 0: [4, 8, 12, 16]                                         │
        │  GPU 1: [4, 8, 12, 16]                                         │
        │  GPU 2: [4, 8, 12, 16]                                         │
        │  GPU 3: [4, 8, 12, 16]                                         │
        └────────────────────────────────────────────────────────────────┘

        This is identical to All-Reduce:
        ┌────────────────────────────────────────────────────────────────┐
        │  [1,2,3,4] + [1,2,3,4] + [1,2,3,4] + [1,2,3,4] = [4,8,12,16]   │
        │                         on all GPUs                            │
        └────────────────────────────────────────────────────────────────┘
```

## Tensor Parallelism in Attention

The attention mechanism is naturally suited for tensor parallelism because of the design of attention heads. In multi-head attention (or multi-query or grouped-query attention), the computation splits across independent heads that only need to be combined at the very end.

```python
def attention_tp(x, rank):
    # Column parallel: each GPU projects for its subset of heads
    q, k, v = W_q[rank](x), W_k[rank](x), W_v[rank](x)
    
    # Local attention computation (no communication needed)
    scores = softmax((q @ k.T) / sqrt(d_head))
    local_out = W_o[rank](scores @ v)
    
    # Single all-reduce to combine partial outputs
    return all_reduce(local_out)
```

As shown in the pseudocode, the Q, K, and V projection matrices use **column parallelism**. Each GPU receives the full input `x` and computes projections for a subset of attention heads (`W_q[rank]`, `W_k[rank]`, `W_v[rank]`). Since attention heads operate independently, no communication is needed during the attention computation itself—each GPU handles its assigned heads in isolation.

The output projection matrix W_o uses **row parallelism**. Each GPU holds rows corresponding to its attention heads' output dimensions (`W_o[rank]`). The partial outputs from each GPU are summed via a single `all_reduce` to produce the final result.

The following diagram illustrates how tensor parallelism distributes the attention computation across four GPUs, with each GPU processing a subset of attention heads independently until the final all-reduce:

```markdown
              Attention with Tensor Parallelism (4 GPUs)
              ═══════════════════════════════════════════

              ┌───────────────────────────────────────────────────┐
              │                   Full Input X                    │
              └───────────────────────────────────────────────────┘
                   │            │            │            │
                   ▼            ▼            ▼            ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ W_q,W_k │ │ W_q,W_k │ │ W_q,W_k │ │ W_q,W_k │
 Column       │   W_v   │ │   W_v   │ │   W_v   │ │   W_v   │
 Parallel     │ (heads  │ │ (heads  │ │ (heads  │ │ (heads  │
              │  0-7)   │ │  8-15)  │ │ 16-23)  │ │ 24-31)  │
              └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
                   │           │           │           │
                   ▼           ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 No comm      │Attention│ │Attention│ │Attention│ │Attention│
 needed       │ compute │ │ compute │ │ compute │ │ compute │
              └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
                   │           │           │           │
                   ▼           ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 Row          │   W_o   │ │   W_o   │ │   W_o   │ │   W_o   │
 Parallel     │(partial)│ │(partial)│ │(partial)│ │(partial)│
              └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
                   │           │           │           │
                   └───────────┴──────┬────┴───────────┘
                                      │
                                 ┌────▼─────┐
                                 │All-Reduce│
                                 └────┬─────┘
                                      │
                                      ▼
                         ┌───────────────────────┐
                         │   Attention Output    │
                         └───────────────────────┘
```

This design requires only a **single all-reduce** at the end of attention. The column-parallel projections feed directly into row-parallel output projection without intermediate communication because the attention computation naturally keeps data separated by heads.


## Tensor Parallelism in the MLP

The MLP layer follows the same column-then-row parallelism pattern as attention, but the motivation differs.

```python
def mlp(x):
    return W_down(activation(W_up(x)) * W_gate(x))
```

**Up and gate projections (column parallelism):** Each GPU stores different columns of `W_up` and `W_gate`, producing a slice of the expanded hidden representation. Since the activation function and element-wise multiplication operate independently on each element, they run locally on each GPU without any communication.

**Down projection (row parallelism):** Each GPU holds the rows of `W_down` that correspond to its slice of the intermediate representation. Multiplying these together produces a partial result, and an all-reduce sums these partials across GPUs to get the final output.

## Tensor Parallelism in the MLP

The MLP layer follows the same column-then-row parallelism pattern as attention, but the motivation differs.

```python
def mlp(x):
    return W_down(activation(W_up(x)) * W_gate(x))
```

**Up and gate projections (column parallelism):** Each GPU stores different columns of `W_up` and `W_gate`, producing a slice of the expanded hidden representation. Since the activation function and element-wise multiplication operate independently on each element, they run locally on each GPU without any communication.

**Down projection (row parallelism):** Each GPU holds the rows of `W_down` that correspond to its slice of the intermediate representation. Multiplying these together produces a partial result, and an all-reduce sums these partials across GPUs to get the final output.

The following diagram illustrates this flow across four GPUs:

```markdown
        MLP with Tensor Parallelism (4 GPUs)
        ════════════════════════════════════

                ┌───────────────────────────────────────────────────┐
                │                    Full Input X                   │
                └───────────────────────────────────────────────────┘
                     │            │            │            │
                     ▼            ▼            ▼            ▼
                ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    Column      │  W_up   │ │  W_up    │ │  W_up    │ │  W_up    │
    Parallel    │ W_gate  │ │ W_gate   │ │ W_gate   │ │ W_gate   │
                │ (cols   │ │ (cols    │ │ (cols    │ │ (cols    │
                │ 0-1023) │ │1024-2047)│ │2048-3071)│ │3072-4095)│
                └────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
                     │            │            │            │
                     ▼            ▼            ▼            ▼
                ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    No comm     │activation│ │activation│ │activation│ │activation│
    needed      │ * gate   │ │ * gate   │ │ * gate   │ │ * gate   │
                └────┬─────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
                     │             │            │            │
                     ▼             ▼            ▼            ▼
                ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    Row         │ W_down  │ │ W_down   │ │ W_down   │ │ W_down   │
    Parallel    │(partial)│ │(partial) │ │(partial) │ │(partial) │
                └────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
                     │            │            │            │
                     └────────────┴─────┬──────┴────────────┘
                                        │
                                  ┌───────────┐
                                  │ All-Reduce│
                                  └─────┬─────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │       MLP Output      │
                            └───────────────────────┘
```

Why must up and gate be column parallel rather than row parallel? The answer lies in the activation function. Activation functions like SiLU or GELU are **element-wise and non-linear**. If we used row parallelism for the up-projection, each GPU would compute a partial sum of the full hidden representation. To apply the activation function correctly, we would need the complete sum first—requiring an all-reduce *before* the activation. By using column parallelism instead, each GPU has the complete values for its slice of the hidden dimension, so activation can be applied locally without any communication.

## Sequence Parallelism for Normalization

The approach so far achieves two all-reduces per transformer block—one for attention and one for the MLP. But redundant work remains: every GPU computes the normalization layers and residual connections identically.

Sequence parallelism eliminates this redundancy by splitting tokens across GPUs for normalization and residual operations, with each GPU handling 1/N of the sequence.

The key insight is twofold: first, an all-reduce can be decomposed into a reduce-scatter followed by an all-gather, allowing us to insert normalization between these two operations; second, normalization (RMS norm) is computed independently for each token, making this split possible.

Here's how it works. After the row-parallel matrix multiplication (in attention or MLP), instead of performing a full all-reduce, we perform a **reduce-scatter**. This sums the partial results but distributes only 1/N of the tokens to each GPU. Each GPU then adds the residual connection and computes normalization on its portion of the sequence. Before the next column-parallel operation (which needs the full input), we perform an **all-gather** to reconstruct the complete sequence.

```markdown
        Sequence Parallelism for Normalization
        ════════════════════════════════════════

        Row-Parallel Output (before communication)
     ┌────────────┬────────────┬────────────┬────────────┐
     │   GPU 0    │   GPU 1    │   GPU 2    │   GPU 3    │
     │  (partial) │  (partial) │  (partial) │  (partial) │
     └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
           │            │            │            │
           └────────────┴─────┬──────┴────────────┘
                              │
                       ┌──────▼───────┐
                       │Reduce-Scatter│
                       └──────┬───────┘
                              │
           ┌────────────┬─────┴─────┬────────────┐
           ▼            ▼           ▼            ▼
     ┌────────────┬────────────┬────────────┬────────────┐
     │  Tokens    │  Tokens    │  Tokens    │  Tokens    │
     │   0-255    │  256-511   │  512-767   │  768-1023  │
     │            │            │            │            │
     │ + Residual │ + Residual │ + Residual │ + Residual │
     └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
     ┌────────────┬────────────┬────────────┬────────────┐
     │  RMSNorm   │  RMSNorm   │  RMSNorm   │  RMSNorm   │
     │  (local)   │  (local)   │  (local)   │  (local)   │
     └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
           │            │            │            │
           └────────────┴─────┬──────┴────────────┘
                              │
                       ┌──────▼──────┐
                       │  All-Gather │
                       └──────┬──────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────────────────┐
     │              Full Sequence (ready for column-parallel)       │
     └──────────────────────────────────────────────────────────────┘
```

Since reduce-scatter + all-gather = all-reduce, we haven't added any communication overhead. We've simply *split* the existing all-reduce and inserted useful computation in the middle. Each GPU now processes only 1/N of the sequence through normalization and residual operations, eliminating redundant computation.


## Putting It All Together

Combining tensor parallelism and sequence parallelism, each transformer block flows as follows. The input sequence is distributed across GPUs (each has 1/N of tokens). RMSNorm is applied locally on each GPU's token slice, then an all-gather reconstructs the full normalized sequence for the column-parallel Q, K, V projections. Attention computation happens independently per GPU on its assigned heads. The row-parallel output projection produces partial results, followed by reduce-scatter to distribute the result and sum the partials. Each GPU adds the residual on its token slice. The pattern repeats for the MLP: RMSNorm on the local slice, all-gather, column-parallel up/gate, local activation and gating, row-parallel down, reduce-scatter, and residual addition.

```markdown
        Complete Transformer Block with Tensor + Sequence Parallelism
        ══════════════════════════════════════════════════════════════

        ┌────────────────────────────────────────────────────────────┐
        │    Sequence-Parallel Input (each GPU has 1/N tokens)       │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │         RMSNorm (sequence parallel, local)                 │
        │              (each GPU: 1/N of tokens)                     │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  All-Gather   │
                              └───────┬───────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │              Column-Parallel Q, K, V                       │
        │       (each GPU: full sequence, subset of heads)           │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                    Attention Compute                       │
        │             (independent per head, no comm)                │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                  Row-Parallel W_o                          │
        │              (each GPU: partial output)                    │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ Reduce-Scatter│
                              └───────┬───────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │            Residual Addition (sequence parallel)           │
        │              (each GPU: 1/N of tokens)                     │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │         RMSNorm (sequence parallel, local)                 │
        │              (each GPU: 1/N of tokens)                     │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  All-Gather   │
                              └───────┬───────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │            Column-Parallel W_up, W_gate                    │
        │       (each GPU: full sequence, subset of dims)            │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │              Activation × Gate (local)                     │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                 Row-Parallel W_down                        │
        │              (each GPU: partial output)                    │
        └─────────────────────────────┬──────────────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ Reduce-Scatter│
                              └───────┬───────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │            Residual Addition (sequence parallel)           │
        │              (each GPU: 1/N of tokens)                     │
        └────────────────────────────────────────────────────────────┘
```

The elegance of this design is remarkable. Every transformer block requires exactly **four point-to-point communications**: two reduce-scatters and two all-gathers. These combine to give the equivalent of two all-reduces—the theoretical minimum for synchronizing parallel matrix multiplications. No redundant computation occurs anywhere. Each GPU performs exactly 1/N of all work: 1/N of matrix multiplications (by parameters), 1/N of attention computations (by heads), and 1/N of normalization (by tokens).


## Practical Considerations

Tensor parallelism's effectiveness depends critically on communication bandwidth. The algorithm shines with fast interconnects like NVLink (up to 900 GB/s on H100) or NVSwitch that enable low-latency, high-bandwidth transfers between GPUs. On slower interconnects like PCIe, communication becomes the bottleneck and other parallelism strategies (like pipeline parallelism) may be more appropriate.

In practice, tensor parallelism is often combined with other techniques. For very large models, you might use tensor parallelism within a node (where GPUs have fast interconnects) and pipeline parallelism or data parallelism across nodes (where network bandwidth is more limited).


## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053) — The original paper introducing tensor parallelism.
- [PyTorch Tensor Parallelism Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — Practical implementation guide.
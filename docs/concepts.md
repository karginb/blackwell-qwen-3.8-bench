# Qwen3.8-27B Local Inference — Concepts

> Concepts collected while getting **Qwen3.8-27B** to run on an **8 GB RTX 5070 Laptop GPU**.
> Every definition is followed by the number we actually measured on this machine, so the
> concept and its real-world magnitude sit together.

---

## 1. The underlying physics: memory, not compute

### Memory bandwidth
Bytes per second a processor can pull from memory. In LLM inference **this is the bottleneck**, not arithmetic throughput.

```
Measured:
  GPU VRAM (read-only) :  353.0 GB/s
  CPU DRAM             :   43.9 GB/s     <- 8x slower
  PCIe (host->device)  :   18.2 GB/s
```

### Arithmetic intensity
Operations performed per byte read from memory. Low means memory-bound, high means compute-bound.

```
Ridge point (this GPU) : ~85 ops/byte
decode                 :   ~6 ops/byte   -> MEMORY bound
prefill (batch 256)    : ~1600 ops/byte  -> COMPUTE bound
```

### Roofline model
Answers "what is the theoretical ceiling on this hardware?" Divide bandwidth by model size:

```
decode speed (tok/s) ~= bandwidth (GB/s) / model size (GB)
```

Measure it rather than assume it: `bench/roofline.py`

### Amdahl's law (as it applies here)
The slow part of a job dominates the whole job. Here, the slow part is the fraction of weights left on the CPU.

The linear model we derived from measurements:

```
time per token = 44.9 ms + 3.29 ms x (layers on CPU)
```

It held within 6% at every measured point. That `3.29 ms/layer` corresponds to 36.7 GB/s of effective CPU bandwidth — 84% of the 43.9 GB/s we measured, which is a good efficiency for llama.cpp.

---

## 2. Two phases: prefill and decode

### Prefill (prompt processing, `pp`)
Reading the input. **All input tokens can be processed in parallel** — read a weight matrix once, multiply it against 256 token vectors at once.

```
matrix x matrix  ->  compute-bound  ->  fast
```

### Decode (text generation, `tg`)
Producing the answer. **Necessarily sequential** — you need token 4 before you can compute token 5. Every weight is re-read for every single token.

```
matrix x vector  ->  memory-bound  ->  slow
```

### Why the gap is so wide
The same bytes read yield 256 tokens of work in prefill and 1 token in decode.

```
Measured (llama-bench, ngl=54):
  pp256 : 393.8 tok/s
  tg64  :  12.87 tok/s     <- 30x slower
```

What the user feels:
```
time to first token = prompt length / pp
each token after    = 1 / tg
```

---

## 3. Memory: weights and caches

### Quantization
Storing each weight in fewer than 16 bits. You trade a little accuracy for a lot of space.

```
Qwen3.8-27B at various precisions:
  BF16 (original)  54.7 GB
  FP8 / INT8       29.0 GB
  Q4_K_M           17.1 GB
  Q3_K_S           12.6 GB
  IQ2_XXS           9.0 GB   <- smallest available
```

Note that even the smallest does not fit 8 GB of VRAM. That single fact is what forced this whole project toward hybrid CPU+GPU inference.

### GGUF
llama.cpp's file format: quantized weights, tokenizer and model config in one file. Each tensor can carry **its own** quantization type.

### Mixed-precision quantization
The label in the filename is an **average**. In reality, sensitive layers get more bits and tolerant layers get fewer.

```
"IQ2_XXS" is really 2.59 bits and mixes 9 different types:

  output head (lm_head)     3.19 bits   <- protected
  linear attention (SSM)    3.08 bits   <- protected
  attention                 2.61 bits
  FFN gate/up + down        2.53 bits   <- sacrificed (64% of the model)
  norm                     32.00 bits   <- untouched
```

**Lesson: do not trust the filename, inspect the file.** `bench/gguf_inspect.py`

Why SSM layers are protected: recurrent state is updated token by token, so quantization error **accumulates**. In classical attention it does not.

Why FFN is sacrificed: it is the largest part of the model (64%), so that is where the space is, and thousands of summed neurons wash out a little noise.

### Importance matrix (imatrix)
Measuring, on real text, how much each weight actually affects the output, then distributing bits accordingly. Types starting with `IQ` use this — there is a large gap between a modern `IQ2_S` and an old naive `Q2_K`.

### KV cache
Attention needs the key (K) and value (V) vectors of every past token. Without caching them, the entire history would be recomputed for every new token.

```
This model: 64 KB per token

   4,096 context  ->  256 MiB
   8,192 context  ->  512 MiB
  32,768 context  ->    2 GiB
```

**The critical point:** the KV cache competes with the weights for the **same VRAM budget**. Extending context does not slow you down directly — it slows you down by stealing space from weights.

```
Measured (at a fixed VRAM budget):
  ngl 52 + 4096 context :  9.87 tok/s
  ngl 50 + 8192 context :  8.68 tok/s   <- -12%
```

### KV cache quantization (`-ctk` / `-ctv`)
Storing the KV cache in 8 bits instead of 16. **Not a speedup in itself** — the dequantization cost exceeds the bandwidth saved. It only pays off if the freed VRAM buys enough extra weight layers.

```
Measured (8K context):
  same ngl=50 : f16 10.25 vs q8_0 9.82   <- q8_0 is SLOWER
  best case   : f16 10.25 vs q8_0 10.48  <- +2.2%, marginal
```

### Recurrent state cache (rs cache)
The state carried by linear-attention layers. **Independent of context length** — fixed size.

```
This model: ~449 MiB (in server mode)
With MTP  : 449 MiB x (draft depth + 1)
```

### Logits buffer
The model produces a score for every word in the vocabulary. Normal generation only needs this for the last token; perplexity needs it for **every** token.

```
This model's vocabulary: 248,320 (multilingual, hence large)

generation :    1 x 248,320 x 4 bytes =    1 MB
perplexity : 2048 x 248,320 x 4 bytes = 2.03 GB   <- blows up VRAM
```

Fix: shrink the batch with `-b 512`.

---

## 4. Architecture

### Attention
Computes how relevant every token is to every other token. Cost grows with the **square** of context length.

### GQA (Grouped Query Attention)
Multiple query heads share one K/V head, shrinking the KV cache.

```
This model: 24 query heads, 4 KV heads  ->  6x saving
```

### Linear attention / SSM (state space model)
Instead of classical attention, carry a **recurrent** state:

```
state <- f(state, new token)
```

Constant memory regardless of context length. But error accumulates (see the quantization note above).

### Hybrid architecture
What this model does: some layers use classical attention, the rest use linear attention.

```
Qwen3.8-27B: 64 layers, pattern "3 linear + 1 full"
  -> 16 full attention   (hold a KV cache)
  -> 48 linear attention (hold a fixed-size state)
```

**The payoff:** the KV cache is 64 KB/token instead of 256 KB — a 4x saving. The compute cost of an 8K context is only -5%.

**The price:** speculative decoding gets expensive, because rolling back a recurrent state requires checkpoints.

### RoPE (Rotary Position Embedding)
Encoding position as a **rotation angle**. Each dimension pair rotates at a different speed:

```
fast-rotating dimensions  ->  local position ("the previous word")
slow-rotating dimensions  ->  global position ("where in the document")
```

### YaRN (Yet another RoPE extensioN)
A method for extending a model's context beyond what it was trained on. It treats frequency bands differently:

```
fast dimensions  ->  leave alone     (preserve local detail)
slow dimensions  ->  interpolate     (let global position stretch)
middle band      ->  smooth ramp
plus an attention temperature correction
```

**Unnecessary for this model:** `rope_theta = 10,000,000` (normally 10,000) means it was trained for 256K context from the start. What YaRN retrofits, Qwen handled during training.

---

## 5. Acceleration techniques

### Flash Attention (`-fa on`)
Computing attention **without ever materializing** the `N x N` matrix. It splits Q/K/V into blocks that fit in on-chip SRAM, computes block by block, and discards the intermediate.

```
GPU memory (VRAM) : 8 GB    but ~353 GB/s
On-chip (SRAM)    : ~200 KB but ~100x faster
```

The mathematical trick is **online softmax**: carry a running maximum and running sum, and rescale the previous partial result when a new block arrives.

**The result is bit-for-bit equivalent**, not an approximation. Memory goes from `O(N^2)` to `O(N)`.

Its real value for us: it is a prerequisite for KV cache quantization in llama.cpp (`-ctk`/`-ctv` require FA).

### Speculative decoding
A fast drafter proposes N tokens; the large model verifies all N in **one forward pass**.

```
normal      : read weights -> 1 token
speculative : read weights -> verify N tokens
```

The premise: verification uses compute that was sitting idle anyway. **Output is mathematically identical.**

### MTP (Multi-Token Prediction)
The drafter is not a separate model but a head **trained into** the main model. In this model it is `blk.64` (the 65th block, ~220 MB).

Advantage: it sees the main model's own hidden state, so its acceptance rate is very high.

```
Measured acceptance:
  depth 1 : 97.7%     <- outstanding (a separate draft model typically gets 60-70%)
  depth 2 : 90.9%
  depth 3 : 84.3%
```

### Draft acceptance rate
The fraction of drafted tokens that survive verification: `draft_n_accepted / draft_n`. This single number decides whether speculative decoding is winning.

### The most important finding of this project
**Speculative decoding's core premise does not hold in hybrid CPU+GPU inference.**

```
On GPU : 93% of compute sits idle  ->  the 2nd token is nearly free
On CPU : AVX2 is already strained  ->  the 2nd token costs ~2x
```

Because our bottleneck is the CPU-resident layers, MTP was a net loss:

```
mechanism : 97.7% acceptance -> working flawlessly
expected  : 8.25 x 1.98 = 16.3 tok/s
measured  : 6.66 tok/s        -> each pass became 2.45x more expensive
```

Three causes: (a) layers were sacrificed to make room for the rs cache, (b) batching is not free on the CPU side, (c) a state checkpoint per draft step.

The literature assumes the model lives entirely in VRAM. Once part of it is on the CPU, the equation inverts.

### Prompt caching (`cache_prompt`)
The server keeps the KV cache of the previous request; if the next request shares a prefix, only the new tokens are processed.

```
Adding a 50-token message to an 8K conversation:
  cache off : process 8050 tokens = 20.6 s
  cache on  : process   50 tokens =  0.13 s    <- 160x
```

**Turn it off in benchmarks** — otherwise everything after the first repetition measures cache hits instead of throughput.

---

## 6. Hardware

### VRAM vs RAM
```
VRAM : GPU memory, fast but small   (8 GB here, ~7.3 usable)
RAM  : system memory, slow but big  (30 GB here)
```

### Hybrid offload (`-ngl`)
Splitting the model between GPU and RAM. This is llama.cpp's decisive capability — vLLM and TensorRT-LLM cannot do it.

```
-ngl 0   -> everything on CPU
-ngl 52  -> 52 layers on GPU, 12 in RAM   <- our optimum
-ngl 65  -> everything on GPU (DOES NOT FIT here)
```

The single largest lever:
```
ngl 20 ->  5.28 tok/s
ngl 56 -> 15.82 tok/s    (llama-bench, empty context)
```

### Compute capability (sm_XX)
The machine-code generation of an NVIDIA GPU. Each generation speaks a different instruction set.

```
sm_86  -> RTX 3000 (Ampere)
sm_89  -> RTX 4000 (Ada)
sm_120 -> RTX 5000 (Blackwell)   <- this card
```

The `a` suffix in `120a` means: also use Blackwell-exclusive, non-forward-compatible instructions.

### PTX JIT
If a binary contains no real machine code for your GPU generation, the driver translates an intermediate representation (PTX) at runtime. Slow startup and no access to newer instructions. **This is why prebuilt binaries underperform on new cards.**

### SIMD (AVX2, AVX-512, AVX-VNNI)
"One instruction, many numbers."

```
AVX2      -> 8 numbers at once
AVX-512   -> 16 at once   (NOT present on this CPU; Intel disabled it this generation)
AVX-VNNI  -> 8-bit integer multiply-accumulate in one instruction  <- what actually
             matters for quantized weights
```

`-march=native` picks these up automatically at build time. Verify in the binary with `objdump | grep vpdpbusd`.

### P-core / E-core (hybrid CPU)
```
i7-14650HX:
  8 physical P-cores x 2 (HT) = 16 logical CPUs   5000-5200 MHz
  8 physical E-cores x 1      =  8 logical CPUs   3700 MHz
                                -----
                                24 logical CPUs
```

E-cores run 28% slower and use a weaker microarchitecture (Gracemont vs Raptor Cove).

### Hyperthreading (SMT)
Duplicating a physical core's **bookkeeping** (registers, program counter) but not its **execution units**. While one thread waits on memory, the other uses the idle units.

Chef analogy: one chef, two order queues. While waiting on the oven, the chef starts the second order. The chef did not multiply — **the waiting gaps got filled**.

**Crucially:** if there is no waiting (compute-bound work), it buys nothing.

```
Measured (llama.cpp defaults to 8 threads here):
  -t  4 :  7.65 tok/s   -23.5%   (not enough parallelism)
  -t  8 :  9.99 tok/s   baseline (8 P-cores, no HT)
  -t 16 : 12.33 tok/s   +23.5%   <- HT WON
  -t 24 :  8.22 tok/s   -17.8%   (E-cores drag)
```

llama.cpp's source says `"hyperthreading isn't useful for linear algebra"` — true for compute-bound work, but the opposite held for our **memory-bound** workload.

Also note: `-t 16` does not mean 16 cores. It means 16 threads on the same 8 physical P-cores.

### Straggler
llama.cpp runs threads in *lockstep*: nobody advances until everyone finishes the current step. One slow core holds up the entire group.

```
-t 24 result: 8.22 tok/s with +/-18.6% spread  <- both slow and unstable
```

### Thermal throttling
A hot chip lowers its clocks. Unavoidable on a laptop.

```
Measured:
  first request : 11.43 tok/s @ 1852 MHz   <- boost clocks
  steady state  :  9.87 tok/s @ 1537 MHz   <- -14%
```

**The steady-state number is the real number.** Boost lasts only the first few seconds.

A curious side effect: the more CPU-bound a configuration, the less the GPU heats up, and the more stable the measurement becomes.
```
IQ2 (GPU-heavy) : spread +/-1.3%, clocks 1530-1837
Q3  (CPU-heavy) : spread +/-0.3%, clocks 1815-1905
```

---

## 7. Engines and tools

### llama.cpp / vLLM / TensorRT-LLM
```
vLLM          -> server, many users. Model must fit ENTIRELY in VRAM
TensorRT-LLM  -> NVIDIA's engine, fastest. Also ENTIRELY in VRAM
llama.cpp     -> single user, CAN SPLIT between GPU and RAM   <- only option at 8 GB
```

### Ollama
A wrapper over llama.cpp. It runs the same engine underneath, but makes the build flags and decisions like `-ngl` for you — conservatively.

### llama.cpp tools
```
llama-cli         -> interactive chat. Not suitable for measurement (single run, no telemetry)
llama-server      -> OpenAI-compatible API. Gives JSON timings + acceptance rate  <- right tool
llama-bench       -> fast sweeps, tight error bars. But no speculative decoding support
llama-perplexity  -> quality measurement. Also has --hellaswag / --multiple-choice
```

**Important trap:** `llama-bench` sizes context to the test (`n_ctx = n_prompt + n_gen`), so it produces **optimistic** numbers.

```
llama-bench -p 256 -n 64  ->  320 token context,  20 MiB KV
llama-server -c 4096      -> 4096 token context, 256 MiB KV   <- 12.8x difference
```

When comparing, `-c`, `-b` and `-ub` must be held constant.

### `--fit-target` (automatic layer fitting)
Letting llama.cpp choose `-ngl` itself. It turns out to be conservative:

```
manual ngl=52 :  9.87 tok/s  -  used 7550 MiB
--fit-target  :  8.89 tok/s  -  used 6908 MiB  (642 MiB left unused)
                 ---------
                 manual is 11% ahead
```

It cannot be combined with `-ngl` — passing both silently disables the fitter.

---

## 8. Measurement methodology

### Perplexity (PPL)
"On average, how many equally likely options was the model torn between?" Lower is better.

```
PPL = exp( -(1/N) x sum log P(actual token | previous tokens) )
```

**Meaningless in isolation**; it only carries information as a comparison — same model, same text, different quantization.

```
Measured (wikitext-2, 100 chunks):
  IQ2_XXS (2.59 bits) : 7.3922 +/- 0.114
  Q3_K_XL (3.93 bits) : 6.6749 +/- 0.101
                        ------
                        10.7% apart, about 7x the error bars -> solid
```

Why PPL rather than a quiz: quantization damage begins by **flattening the distribution**. The model may still pick the right word, but with 55% confidence instead of 90%. PPL catches that; a quiz does not.

**Its limit:** it measures next-word prediction on English prose. It does not measure fragility in code or tool-calling, where the failure mode is not "a duller answer" but "unparseable JSON".

### Median vs mean
The median is immune to a single slow run. Benchmarks should report the median.

### Spread
`(max - min) / 2 / median`. A measure of how much to distrust the number. **A result whose spread exceeds its margin has proven nothing.**

```
Effect of fixing the methodology:
  llama-cli, no repeats     : +/-14.8%  -> produced ngl=48 < ngl=46 (physically impossible)
  llama-server + discard    : +/-0.5%   -> trustworthy
```

### Discarding warm-up reps (`--discard`)
The first requests run on boost clocks. Drop them to measure the steady state.

```
raw: [11.43, 9.92, 9.88, 9.87, 9.79]
      \boost/  \--- steady state ---/
```

### Benchmark hygiene — the rules
1. Every repetition must do the **same work from scratch** (`cache_prompt` off)
2. Compared configurations must share the **same parameters** (`-c`, `-b`, `-ub`)
3. **Cool down between configurations** — otherwise thermal decay masquerades as a performance difference
4. Measure **in the same run** — different runs have different thermal conditions
5. Never measure throughput with a short prompt (fixed overheads dominate)
6. **Match what you measure to what you will deploy**

---

## 9. This project's numbers

### Hardware
```
GPU  : RTX 5070 Laptop, 8151 MiB VRAM (~7.3 usable), sm_120
CPU  : i7-14650HX, 8 P-cores + 8 E-cores, AVX2 + AVX-VNNI (no AVX-512)
RAM  : 30 GB DDR5
```

### Model
```
Qwen3.8-27B : 27.32B parameters, 64 layers + 1 MTP block
hybrid      : 16 full attention + 48 linear attention
head_dim    : 256 (unusual; most models use 64-128)
GQA         : 24 query / 4 KV heads
vocabulary  : 248,320 (multilingual)
context     : 262,144 native (rope_theta = 10M)
KV cost     : 64 KB/token
```

### Layer sizes
```
IQ2_XXS : 121 MB/layer  ->  ngl 52 fits
Q3_K_XL : 183 MB/layer  ->  ngl 34 fits
```

### Best configuration (realistic, steady state)
```bash
llama-server -m Qwen3.8-27B-UD-IQ2_XXS.gguf -ngl 52 -fa on -c 4096 -t 16
-> 12.33 tok/s
```

### Ground covered
```
first run (ngl 20)     :  5.28 tok/s
+ ngl optimization     :  9.99
+ hyperthreading       : 12.33      <- 2.3x
```

### Decisions
```
-fa on        ->  always on
-ngl          ->  tune by hand, do not trust --fit-target (11% gap)
-t            ->  logical P-core count (not physical!) = 16
-c            ->  only as much as you need; 4K->8K costs 12%
KV quant      ->  only if it buys extra layers
MTP           ->  DO NOT enable under hybrid offload (net loss even with a
                  flawless 97.7% acceptance rate)
quant choice  ->  IQ2 vs Q3: 11% quality for 49% speed -> IQ2 for general use
```

### Still open
- Task-level quality (`--hellaswag`, `--multiple-choice`) — perplexity does not cover code or tool-calling
- `autotune.py` — a tool that reads the hardware and derives these settings automatically
- Comparison against a small model that fits entirely in VRAM (9B @ 4-bit)

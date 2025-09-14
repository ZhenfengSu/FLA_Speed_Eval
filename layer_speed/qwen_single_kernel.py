import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Device helper (NPU/CUDA/CPU)
# --------------------------
def has_npu():
    return hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0

def current_device():
    if has_npu():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def sync_device(dev: torch.device):
    if dev.type == "npu":
        torch.npu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()
    else:
        pass

def autocast_context(dev: torch.device, dtype: torch.dtype, enabled=True):
    if not enabled:
        return torch.autocast(device_type=dev.type, enabled=False)
    return torch.autocast(device_type=dev.type, dtype=dtype, enabled=True)

# --------------------------
# Config（仅 prefill，无 KV cache）
# --------------------------
class Config:
    # 模型形状
    batch_size = 1
    seq_len = 1024
    hidden_dim = 4096
    n_heads = 32
    ffw_mult = 8/3
    bias = False

    # 精度与设备
    dtype = "bf16"  # "fp32" | "fp16" | "bf16"
    deterministic = False
    use_compile = False  # 依据环境决定是否可用

    # 运行设置
    warmup_steps = 10
    measure_steps = 30

    # 训练/反向传播
    run_backward = True
    grad_accum_steps = 1
    learning_rate = 1e-3
    weight_decay = 0.0

    # dropout（训练时可设>0）
    attn_dropout = 0.0
    ffn_dropout = 0.0

    # 门控 MLP 配置
    gated_mlp_activation = "swish"  # "swish" (SwiGLU) | "gelu" (GEGLU)

cfg = Config()

# --------------------------
# Utils
# --------------------------
def get_dtype():
    if cfg.dtype == "fp16":
        return torch.float16
    if cfg.dtype == "bf16":
        return torch.bfloat16
    return torch.float32

def setup_backend(dev: torch.device):
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not cfg.deterministic
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

def training_mode():
    return cfg.run_backward

# --------------------------
# RMSNorm
# --------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

# --------------------------
# 线性层
# --------------------------
def linear(in_f, out_f, bias: bool):
    return nn.Linear(in_f, out_f, bias=bias)

# --------------------------
# 单个 Attention kernel（把你的算子实现放到这里）
# q/k/v: [B, nH, S, d] -> out: [B, nH, S, d]
# 默认给出 SDPA 示例；可替换为自研内核
# --------------------------
def attention_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True):
    # 示例：PyTorch SDPA。若替换为自研算子，确保接口和返回形状一致。
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training_mode() else 0.0,
        is_causal=is_causal
    )

# 如需一个“参考实现”来对照你的自研版本，可以临时把上面替换为手写实现：
# def attention_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True):
#     d = q.size(-1)
#     attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
#     if attn_mask is not None:
#         attn_scores = attn_scores + attn_mask
#     if is_causal and (attn_mask is None):
#         S_q = q.size(-2); S_k = k.size(-2)
#         i = torch.arange(S_q, device=q.device).unsqueeze(-1)
#         j = torch.arange(S_k, device=q.device).unsqueeze(0)
#         causal = (j > i)
#         attn_scores = attn_scores.masked_fill(causal, float("-inf"))
#     attn_probs = torch.softmax(attn_scores, dim=-1)
#     if dropout_p > 0 and training_mode():
#         attn_probs = F.dropout(attn_probs, p=dropout_p)
#     return torch.matmul(attn_probs, v)

# --------------------------
# Multi-Head Attention
# --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, bias=False, attn_dropout=0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.proj = linear(hidden_dim, hidden_dim, bias=bias)
        self.attn_dropout = attn_dropout

    def forward(self, x, attn_mask=None, is_causal=True):
        B, S, H = x.shape
        qkv = self.qkv(x)  # [B, S, 3H]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nH, S, d]
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = attention_kernel(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.proj(out)
        return out

# --------------------------
# 门控 MLP（Hadamard 逐元素乘）：SwiGLU / GEGLU
# y = W_down( act(W_gate(x)) ⊙ W_up(x) )
# --------------------------
class GatedFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffw_mult=4, bias=False, dropout=0.0, activation="swish"):
        super().__init__()
        inner = int(ffw_mult * hidden_dim)
        self.up = linear(hidden_dim, inner, bias=bias)    # W_up
        self.gate = linear(hidden_dim, inner, bias=bias)  # W_gate
        self.down = linear(inner, hidden_dim, bias=bias)  # W_down
        self.dropout = nn.Dropout(dropout)

        act = activation.lower()
        if act == "swish":
            self.act = nn.SiLU()  # SwiGLU
        elif act == "gelu":
            self.act = nn.GELU(approximate="tanh")  # GEGLU
        else:
            raise ValueError(f"Unsupported gated activation: {activation}. Use 'swish' or 'gelu'.")

    def forward(self, x):
        up = self.up(x)                # [B, S, inner]
        gate = self.act(self.gate(x))  # [B, S, inner]
        y = gate * up                  # Hadamard product
        y = self.down(y)               # [B, S, H]
        return self.dropout(y)

# --------------------------
# 单层 Transformer（Pre-RMSNorm + MHA + 残差 + Pre-RMSNorm + 门控MLP + 残差）
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffw_mult=4, bias=False, attn_dropout=0.0, ffn_dropout=0.0, gated_activation="swish"):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, n_heads, bias=bias, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = GatedFeedForward(hidden_dim, ffw_mult=ffw_mult, bias=bias, dropout=ffn_dropout, activation=gated_activation)

    def forward(self, x, attn_mask=None, is_causal=True):
        h = self.norm1(x)
        attn_out = self.attn(h, attn_mask=attn_mask, is_causal=is_causal)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.mlp(h2)
        return x

# --------------------------
# 基准：前向（prefill）
# --------------------------
@torch.no_grad()
def benchmark_infer(block, x, steps_warmup, steps_measure, dev):
    for _ in range(steps_warmup):
        _ = block(x)
    sync_device(dev)

    times = []
    for _ in range(steps_measure):
        t0 = time.time()
        _ = block(x)
        sync_device(dev)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std

# --------------------------
# 基准：训练（prefill 前+反+step）
# --------------------------
def benchmark_train(block, x, steps_warmup, steps_measure, optimizer, dev, use_amp=True):
    block.train()
    loss_fn = nn.MSELoss()
    target = torch.zeros_like(x)

    # 预热
    for _ in range(steps_warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(dev, get_dtype(), enabled=use_amp):
            out = block(x)
            loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    sync_device(dev)

    times_fwd = []
    times_step = []
    for _ in range(steps_measure):
        optimizer.zero_grad(set_to_none=True)
        t0_fwd = time.time()

        total_loss = 0.0
        for _acc in range(cfg.grad_accum_steps):
            with autocast_context(dev, get_dtype(), enabled=use_amp):
                out = block(x)
                loss = loss_fn(out, target) / cfg.grad_accum_steps
            loss.backward()
            total_loss += float(loss.detach())

        sync_device(dev)
        t1_fwd = time.time()

        optimizer.step()
        sync_device(dev)
        t2 = time.time()

        times_fwd.append((t1_fwd - t0_fwd) * 1000.0)
        times_step.append((t2 - t0_fwd) * 1000.0)

    mean_fwd = sum(times_fwd) / len(times_fwd)
    std_fwd = (sum((t - mean_fwd) ** 2 for t in times_fwd) / len(times_fwd)) ** 0.5
    mean_step = sum(times_step) / len(times_step)
    std_step = (sum((t - mean_step) ** 2 for t in times_step) / len(times_step)) ** 0.5
    return (mean_fwd, std_fwd), (mean_step, std_step)

# --------------------------
# 主函数
# --------------------------
def main():
    dev = current_device()
    setup_backend(dev)
    dt = get_dtype()

    B = cfg.batch_size
    S = cfg.seq_len
    H = cfg.hidden_dim
    nH = cfg.n_heads
    assert H % nH == 0, "hidden_dim 必须能被 heads 整除"

    x = torch.randn(B, S, H, device=dev, dtype=dt, requires_grad=cfg.run_backward)

    block = TransformerBlock(
        hidden_dim=H,
        n_heads=nH,
        ffw_mult=cfg.ffw_mult,
        bias=cfg.bias,
        attn_dropout=cfg.attn_dropout if cfg.run_backward else 0.0,
        ffn_dropout=cfg.ffn_dropout if cfg.run_backward else 0.0,
        gated_activation=cfg.gated_mlp_activation
    ).to(device=dev, dtype=dt)

    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            block = torch.compile(block, mode="max-autotune")
        except Exception as e:
            print(f"torch.compile 跳过（设备={dev}, 原因: {e})")

    print("===== Prefill Benchmark Config =====")
    print(f"device          : {dev}")
    print(f"dtype           : {dt}")
    print(f"batch_size      : {B}")
    print(f"seq_len         : {S}  (prefill only, no KV cache)")
    print(f"hidden_dim      : {H}")
    print(f"heads           : {nH}")
    print(f"ffw_mult        : {cfg.ffw_mult}")
    print(f"bias            : {cfg.bias}")
    print(f"deterministic   : {cfg.deterministic}")
    print(f"warmup/meas     : {cfg.warmup_steps}/{cfg.measure_steps}")
    print(f"run_backward    : {cfg.run_backward}")
    print(f"grad_accum      : {cfg.grad_accum_steps}")
    print(f"gated_mlp_act   : {cfg.gated_mlp_activation}")
    print("============================")

    if not cfg.run_backward:
        block.eval()
        mean, std = benchmark_infer(block, x,
                                    steps_warmup=cfg.warmup_steps, steps_measure=cfg.measure_steps, dev=dev)
        print("\n===== Inference (Prefill) =====")
        print(f"Kernel (single) : mean={mean:.3f} ms, std={std:.3f} ms")
    else:
        params = [p for p in block.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        use_amp = (dt in (torch.float16, torch.bfloat16)) and (dev.type in ("cuda", "npu"))

        (fwd_mean, fwd_std), (step_mean, step_std) = benchmark_train(
            block, x,
            steps_warmup=cfg.warmup_steps, steps_measure=cfg.measure_steps,
            optimizer=optimizer, dev=dev, use_amp=use_amp
        )

        print("\n===== Training (Prefill) =====")
        print("Forward only (ms/iter):")
        print(f"  mean={fwd_mean:.3f}, std={fwd_std:.3f}")
        print("\nForward+Backward+Step (ms/iter):")
        print(f"  mean={step_mean:.3f}, std={step_std:.3f}")

if __name__ == "__main__":
    main()
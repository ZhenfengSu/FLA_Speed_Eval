import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Device helpers (NPU/CUDA/CPU) ==========
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

# ========== Config (prefill only, single kernel) ==========
class Config:
    # 模型
    batch_size = 4
    seq_len = 1024
    hidden_dim = 4096
    n_heads = 32          # 查询头数（Llama 3：n_heads）
    n_kv_heads = 8        # KV头数（Llama 3 使用 GQA：n_kv_heads | 必须整除 n_heads）
    bias = False

    # MLP
    inner_mult = 4/3      # Llama 3 常用 4/3 * hidden_dim（约 1.333）
    mlp_dropout = 0.0

    # 精度与运行
    dtype = "bf16"        # "fp32" | "fp16" | "bf16"
    deterministic = False
    use_compile = False
    warmup_steps = 10
    measure_steps = 30

    # 训练设置
    run_backward = True
    grad_accum_steps = 1
    learning_rate = 1e-3
    weight_decay = 0.0

    # 注意力
    attn_dropout = 0.0

    # RoPE 配置（Llama 3）
    rope_theta = 1e4
    rope_base = 1e4       # 同 theta
    rope_scaling = 1.0    # 1.0 表示不缩放；可支持长上下文时的比例缩放
    rope_partial = 1.0    # 1.0 表示全维应用RoPE；可设为 <1 仅对一部分维度做RoPE

cfg = Config()

# ========== Utils ==========
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

def linear(in_f, out_f, bias=False):
    return nn.Linear(in_f, out_f, bias=bias)

# ========== RMSNorm ==========
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

# ========== RoPE (Llama-style rotary embeddings) ==========
def build_rope_cache(seqlen, head_dim, device, dtype, theta=1e4, partial_ratio=1.0):
    d = head_dim
    d_rope = int(d * partial_ratio)
    d_rope = max(2, d_rope - d_rope % 2)  # 偶数
    d_pass = d - d_rope
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_rope, 2, device=device, dtype=torch.float32) / d_rope))
    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [S, d_rope/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [S, d_rope]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [S, d_rope]
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    return cos, sin, d_rope, d_pass

def apply_rope(x, cos, sin, d_rope, d_pass):
    # x: [B, nH, S, d]
    if d_rope == 0:
        return x
    x_rope = x[..., :d_rope]
    x_pass = x[..., d_rope:]
    x1, x2 = x_rope[..., ::2], x_rope[..., 1::2]  # [B,nH,S,d_rope/2]
    cos_ = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,d_rope]
    sin_ = sin.unsqueeze(0).unsqueeze(0)
    cos1, cos2 = cos_[..., ::2], cos_[..., 1::2]
    sin1, sin2 = sin_[..., ::2], sin_[..., 1::2]
    # 经典旋转：[(x1, x2)] -> [(x1*cos - x2*sin), (x1*sin + x2*cos)]
    y1 = x1 * cos1 - x2 * sin1
    y2 = x1 * sin2 + x2 * cos2
    y = torch.empty_like(x_rope)
    y[..., ::2] = y1
    y[..., 1::2] = y2
    if d_pass > 0:
        y = torch.cat([y, x_pass], dim=-1)
    return y

# ========== 单个注意力内核（将你的算子放在这里） ==========
def attention_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True):
    # 输入 q/k/v: [B, nH, S, d]（这里 nH == n_heads after expand for GQA）
    # 返回:      [B, nH, S, d]
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training_mode() else 0.0,
        is_causal=is_causal
    )

# ========== Llama 3 Attention with GQA + RoPE ==========
class Llama3Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, bias=False, attn_dropout=0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = linear(hidden_dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = linear(hidden_dim, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = linear(hidden_dim, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = linear(n_heads * self.head_dim, hidden_dim, bias=bias)
        self.attn_dropout = attn_dropout

    def repeat_kv(self, x, n_rep):
        # x: [B, n_kv, S, d] -> [B, n_kv*n_rep, S, d]
        if n_rep == 1:
            return x
        B, n_kv, S, d = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv, n_rep, S, d).reshape(B, n_kv * n_rep, S, d)
        return x

    def forward(self, x, cos, sin, d_rope, d_pass, attn_mask=None, is_causal=True):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # [B,nH,S,d]
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B,nKv,S,d]
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B,nKv,S,d]

        # RoPE
        q = apply_rope(q, cos, sin, d_rope, d_pass)
        k = apply_rope(k, cos, sin, d_rope, d_pass)

        # GQA: 扩展 KV 到查询头
        n_rep = self.n_heads // self.n_kv_heads
        k = self.repeat_kv(k, n_rep)  # [B,nH,S,d]
        v = self.repeat_kv(v, n_rep)  # [B,nH,S,d]

        # 注意力（单一 kernel）
        out = attention_kernel(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, S, H)  # [B,S,H]
        out = self.o_proj(out)
        return out

# ========== Llama 3 MLP: SwiGLU ==========
class Llama3MLP(nn.Module):
    def __init__(self, hidden_dim, inner_mult=4/3, bias=False, dropout=0.0):
        super().__init__()
        inner = int(round(inner_mult * hidden_dim))
        self.gate_proj = linear(hidden_dim, inner, bias=bias)
        self.up_proj   = linear(hidden_dim, inner, bias=bias)
        self.down_proj = linear(inner, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()  # SwiGLU

    def forward(self, x):
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        y = gate * up
        y = self.down_proj(y)
        return self.dropout(y)

# ========== Llama 3 Block (Pre-RMSNorm + Attn + Residual + Pre-RMSNorm + MLP + Residual) ==========
class Llama3Block(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, bias=False, attn_dropout=0.0, mlp_inner_mult=4/3, mlp_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = Llama3Attention(hidden_dim, n_heads, n_kv_heads, bias=bias, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = Llama3MLP(hidden_dim, inner_mult=mlp_inner_mult, bias=bias, dropout=mlp_dropout)

    def forward(self, x, cos, sin, d_rope, d_pass, attn_mask=None, is_causal=True):
        h = self.norm1(x)
        attn_out = self.attn(h, cos, sin, d_rope, d_pass, attn_mask=attn_mask, is_causal=is_causal)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.mlp(h2)
        return x

# ========== Mask helper (causal lower-tri) ==========
def causal_mask(S, device, dtype):
    m = torch.ones((S, S), device=device, dtype=torch.bool).triu(1)
    mask = torch.zeros((S, S), device=device, dtype=dtype).masked_fill(m, float("-inf"))
    return mask  # [S,S] broadcast to [B,nH,S,S] by SDPA

# ========== Bench: inference ==========
@torch.no_grad()
def benchmark_infer(block, x, cos, sin, d_rope, d_pass, steps_warmup, steps_measure, dev):
    mask = causal_mask(x.size(1), device=x.device, dtype=x.dtype)
    for _ in range(steps_warmup):
        _ = block(x, cos, sin, d_rope, d_pass, attn_mask=mask, is_causal=False)  # 显式mask
    sync_device(dev)

    times = []
    for _ in range(steps_measure):
        t0 = time.time()
        _ = block(x, cos, sin, d_rope, d_pass, attn_mask=mask, is_causal=False)
        sync_device(dev)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std

# ========== Bench: train (fwd+bwd+step) ==========
def benchmark_train(block, x, cos, sin, d_rope, d_pass, steps_warmup, steps_measure, optimizer, dev, use_amp=True):
    block.train()
    loss_fn = nn.MSELoss()
    target = torch.zeros_like(x)
    mask = causal_mask(x.size(1), device=x.device, dtype=x.dtype)

    # warmup
    for _ in range(steps_warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(dev, get_dtype(), enabled=use_amp):
            out = block(x, cos, sin, d_rope, d_pass, attn_mask=mask, is_causal=False)
            loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    sync_device(dev)

    times_fwd, times_step = [], []
    for _ in range(steps_measure):
        optimizer.zero_grad(set_to_none=True)
        t0_fwd = time.time()

        total_loss = 0.0
        for _acc in range(cfg.grad_accum_steps):
            with autocast_context(dev, get_dtype(), enabled=use_amp):
                out = block(x, cos, sin, d_rope, d_pass, attn_mask=mask, is_causal=False)
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

# ========== Main ==========
def main():
    dev = current_device()
    setup_backend(dev)
    dt = get_dtype()

    B, S, H = cfg.batch_size, cfg.seq_len, cfg.hidden_dim
    nH, nKv = cfg.n_heads, cfg.n_kv_heads
    assert H % nH == 0, "hidden_dim 必须能被 n_heads 整除"
    assert nH % nKv == 0, "n_heads 必须能被 n_kv_heads 整除"

    x = torch.randn(B, S, H, device=dev, dtype=dt, requires_grad=cfg.run_backward)

    # RoPE cache
    cos, sin, d_rope, d_pass = build_rope_cache(S, head_dim=H // nH, device=dev, dtype=dt,
                                                theta=cfg.rope_theta, partial_ratio=cfg.rope_partial)

    block = Llama3Block(
        hidden_dim=H,
        n_heads=nH,
        n_kv_heads=nKv,
        bias=cfg.bias,
        attn_dropout=cfg.attn_dropout,
        mlp_inner_mult=cfg.inner_mult,
        mlp_dropout=cfg.mlp_dropout
    ).to(device=dev, dtype=dt)

    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            block = torch.compile(block, mode="max-autotune")
        except Exception as e:
            print(f"torch.compile 跳过（设备={dev}, 原因: {e})")

    print("===== Prefill Llama3 Benchmark Config =====")
    print(f"device          : {dev}")
    print(f"dtype           : {dt}")
    print(f"batch_size      : {B}")
    print(f"seq_len         : {S} (prefill only)")
    print(f"hidden_dim      : {H}")
    print(f"heads (q)       : {nH}")
    print(f"kv heads        : {nKv}")
    print(f"head_dim        : {H//nH}")
    print(f"inner_mult      : {cfg.inner_mult} (~{int(round(cfg.inner_mult*H))})")
    print(f"attn_dropout    : {cfg.attn_dropout}")
    print(f"mlp_dropout     : {cfg.mlp_dropout}")
    print(f"rope_theta      : {cfg.rope_theta}, partial={cfg.rope_partial}")
    print(f"warmup/meas     : {cfg.warmup_steps}/{cfg.measure_steps}")
    print(f"run_backward    : {cfg.run_backward}")
    print(f"grad_accum      : {cfg.grad_accum_steps}")
    print("============================")

    if not cfg.run_backward:
        block.eval()
        mean, std = benchmark_infer(block, x, cos, sin, d_rope, d_pass,
                                    steps_warmup=cfg.warmup_steps,
                                    steps_measure=cfg.measure_steps, dev=dev)
        print("\n===== Inference (Prefill, Llama3) =====")
        print(f"Kernel (single) : mean={mean:.3f} ms, std={std:.3f} ms")
    else:
        params = [p for p in block.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        use_amp = (dt in (torch.float16, torch.bfloat16)) and (dev.type in ("cuda", "npu"))
        (fwd_mean, fwd_std), (step_mean, step_std) = benchmark_train(
            block, x, cos, sin, d_rope, d_pass,
            steps_warmup=cfg.warmup_steps, steps_measure=cfg.measure_steps,
            optimizer=optimizer, dev=dev, use_amp=use_amp
        )
        print("\n===== Training (Prefill, Llama3) =====")
        print("Forward only (ms/iter):")
        print(f"  mean={fwd_mean:.3f}, std={fwd_std:.3f}")
        print("\nForward+Backward+Step (ms/iter):")
        print(f"  mean={step_mean:.3f}, std={step_std:.3f}")

if __name__ == "__main__":
    main()
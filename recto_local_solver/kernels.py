"""GPU KERNELS for the work PyTorch runs as thousands of small launches or through ill-suited library calls: the SGM path
aggregation, the coarse level's message passing, the slope scan's semblance stack, the separable Gaussian filters
(single-channel convolutions, slow in cuDNN), the structure tensor's eigenvector iterations, the CT slab's trilinear
sampling straight from the GPU chunk cache, the sheared slabs' depth resampling, the fibre tie's
products over lags and depth shifts, the fibre tracker's Viterbi steps and back-tracking, the 3-D streamline
median of the fibre step and the fibre edges' dynamic programming.

They are CUDA C compiled at run time by NVRTC (CuPy RawKernel) for the GPU in use and cached by CuPy, so any NVIDIA
GPU works. They are used on an NVIDIA GPU with PyTorch's CUDA build and CuPy (RECTO_KERNELS=0: never); otherwise (an
AMD / ROCm or Apple GPU, the CPU), or when a kernel fails to compile, the PyTorch code they replace runs. With RECTO_KERNELS_CHECK=1 the first call of each kernel
in a process is also checked against that PyTorch code on the same inputs (a mismatch beyond the tolerance switches the
kernel off, with a line in the log); RECTO_KERNELS_CHECK=name,name checks those kernels only."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

OPTS = ("--std=c++14", "--fmad=false")        # no fused multiply-add: the PyTorch kernels' rounding where it matters
_K = {}
_STATE = {"ok": None}
_CHECKED = {}


def log(msg):
    print(f"kernels: {msg}", file=sys.stderr, flush=True)


def available() -> bool:
    """the kernels run: an NVIDIA GPU with PyTorch's CUDA build (not ROCm / HIP, which also reports torch.cuda) and CuPy,
    and RECTO_KERNELS is not 0. Otherwise every step runs its PyTorch code (any device: the same algorithm, slower)."""
    if _STATE["ok"] is None:
        from . import volume as qc
        ok = (os.environ.get("RECTO_KERNELS", "1") != "0" and qc.DEV.startswith("cuda") and torch.cuda.is_available()
              and torch.version.cuda is not None and getattr(torch.version, "hip", None) is None)
        if not ok and os.environ.get("RECTO_KERNELS", "1") != "0":
            log(f"no NVIDIA CUDA device for the kernels (device {qc.DEV}, torch {torch.__version__}); the PyTorch code runs")
        if ok:
            try:
                import cupy
                cupy.cuda.runtime.getDeviceCount()
            except Exception as e:                                          # no CuPy / no device: PyTorch code
                log(f"CuPy not usable ({type(e).__name__}: {str(e)[:120]}); the PyTorch code runs")
                ok = False
        _STATE["ok"] = ok
    return _STATE["ok"]


def _kernel(name, key=None, opts=OPTS):
    """the compiled kernel `name` (SOURCES[name]; key: a variant of it, its source SOURCES[name](key))."""
    k = (name, key)
    if k not in _K:
        import cupy
        _K[k] = cupy.RawKernel(SOURCES[name]() if key is None else SOURCES[name](key), name, options=opts)
    return _K[k]


def _cp(t):
    """a kernel argument: the tensor, contiguous (held by the argument list until the launch; passed as a pointer)."""
    return t.contiguous()


def _launch(name, grid, block, args, shared=0, key=None, opts=OPTS):
    """the kernel on PyTorch's current stream and device; tensors are passed as raw device pointers (the launch is
    ordered on the stream, so the memory stays valid for it as for PyTorch's own kernels)."""
    import cupy
    a = [np.uint64(x.data_ptr()) if isinstance(x, torch.Tensor) else x for x in args]
    k = _kernel(name, key, opts)
    s = torch.cuda.current_stream().cuda_stream
    if s == 0 and torch.cuda.current_device() == 0:                     # (CuPy's default: the same legacy stream, device 0)
        k(grid, block, a, shared_mem=shared)
    else:
        with cupy.cuda.Device(torch.cuda.current_device()), cupy.cuda.ExternalStream(s):
            k(grid, block, a, shared_mem=shared)


def run(name, fast, ref, tol, mean=False):
    """fast() with the kernel, ref() the PyTorch code; the first call per process runs both and compares (max |diff|
    over the outputs <= tol * max |ref|, or with mean=True the mean |diff|: for a median whose samples may tie
    differently; else the kernel is switched off). Returns the result."""
    if not available() or _CHECKED.get(name) is False:
        return ref()
    try:
        out = fast()
    except Exception as e:
        log(f"{name}: kernel failed ({type(e).__name__}: {str(e)[:200]}); the PyTorch code runs")
        _CHECKED[name] = False
        return ref()
    chk = os.environ.get("RECTO_KERNELS_CHECK", "0")                    # 1: every kernel; or a list of kernel names
    if name not in _CHECKED and (chk == "1" or name in chk.split(",")):
        r = ref()
        a, b = (out if isinstance(out, (tuple, list)) else (out,)), (r if isinstance(r, (tuple, list)) else (r,))
        red = (lambda z: z.mean()) if mean else (lambda z: z.max())
        diff = max(float(red((x.double() - y.double()).abs())) if x.numel() else 0.0 for x, y in zip(a, b))
        scale = max(float(y.double().abs().max()) if y.numel() else 0.0 for y in b)
        _CHECKED[name] = diff <= tol * max(scale, 1e-12)
        log(f"{name}: against the PyTorch code {'mean' if mean else 'max'} |diff| {diff:.3g} (scale {scale:.3g}) -> "
            f"{'kernel used' if _CHECKED[name] else 'kernel OFF, the PyTorch code runs'}")
        return out if _CHECKED[name] else r
    return out


def _threads(n, cap=256):
    """threads per block: the next power of two >= n (>= 32, <= cap; the block reductions halve it)"""
    t = 32
    while t < n and t < cap:
        t *= 2
    return t


# ------------------------------------------------------------------------------------------------ SGM
def sgm_dip(C, DY, DX, DW, lidx, D, P1, sct, cap, dw, lam):
    """the 8-path aggregation of sheetsolve.sgm with the lamination-following pairwise cost computed in the kernel:
    C [NL, H, W]; DY, DX, DW [K, H, W] the dips, lidx [NL] their depth index per label. Returns the 8 path costs
    [8, NL, H, W] (sheetsolve.sgm's order: columns +x with rows 0 / +1 / -1, columns -x the same, rows +y, rows -y).
    Every path is a set of independent chains (rows, diagonals or columns): ONE launch, a block per (chain, path)
    stepping along its chain (per cell the arithmetic of one launch per column / row)."""
    NL, H, W = C.shape
    L = torch.empty((8, NL, H, W), device=C.device, dtype=torch.float32)
    nt = _threads(NL)
    _launch("sgm_dip", (H + W - 1, 8), (nt,), [_cp(C), _cp(DY), _cp(DX), _cp(DW), _cp(lidx.int()), _cp(L), np.int32(NL), np.int32(H),
                                               np.int32(W), np.int32(D), np.float32(P1), np.float32(sct), np.float32(cap),
                                               np.float32(dw), np.float32(lam)], shared=4 * (2 * NL + nt))
    return L


def _src_sgm():
    return r"""
#define BIG 1e4f
extern "C" __global__ void sgm_dip(const float* __restrict__ C, const float* __restrict__ DY, const float* __restrict__ DX,
    const float* __restrict__ DW, const int* __restrict__ lidx, float* __restrict__ L,
    const int NL, const int H, const int W, const int D, const float P1, const float sct, const float cap, const float dw,
    const float lam)
{
  extern __shared__ float sh[];                 // [NL] best per label, [blockDim] the minimum, [NL] the previous step's
  float* red = sh + NL;                         // path costs (as written to L: read from here, not from L)
  float* prev = red + blockDim.x;
  const int p = blockIdx.y, fam = p < 6 ? 0 : 1;
  int xdir = 0, sy = 0, ydir = 0, c = blockIdx.x, steps;
  if (fam == 0) {                               // the six paths along the columns: chain c = a row (sy 0) or a diagonal
    xdir = p < 3 ? 1 : -1; const int r = p % 3; sy = r == 0 ? 0 : (r == 1 ? 1 : -1);
    if (sy == 0 && c >= H) return;
    if (sy == 1) c -= W - 1;                    // the row at step 0: i = c + sy t
    steps = W;
  } else {                                      // the two paths along the rows: chain c = a column
    ydir = p == 6 ? 1 : -1;
    if (c >= W) return;
    steps = H;
  }
  const size_t HW = (size_t)H * W, base = (size_t)p * NL * HW;
  const float oy = fam == 0 ? (float)(-sy) : (float)(-ydir), ox = fam == 0 ? (float)(-xdir) : 0.0f;
  for (int t = 0; t < steps; t++) {
    int i, x, ip, xp;
    if (fam == 0) { x = xdir == 1 ? t : W - 1 - t; i = c + sy * t; ip = i - sy; xp = x - xdir; }
    else { x = c; i = ydir == 1 ? t : H - 1 - t; ip = i - ydir; xp = x; }
    if (i < 0 || i >= H) continue;              // (the chain outside the grid at this step: the whole block)
    __syncthreads();                            // (the previous step's costs written)
    if (t == 0) {
      for (int l = threadIdx.x; l < NL; l += blockDim.x) { const float c0 = C[l * HW + (size_t)i * W + x]; L[base + l * HW + (size_t)i * W + x] = c0; prev[l] = c0; }
      continue;
    }
    const bool rin = ip >= 0 && ip < H;
    const int ic = min(max(ip, 0), H - 1), xc = min(max(xp, 0), W - 1);
    float lmin = 3.4e38f;
    for (int l = threadIdx.x; l < NL; l += blockDim.x) {
      const size_t o = (size_t)lidx[l] * HW;
      const float DYl = DY[o + (size_t)i * W + x], DXl = DX[o + (size_t)i * W + x], Wl = DW[o + (size_t)i * W + x];
      const float DYn = DY[o + (size_t)ic * W + xc], DXn = DX[o + (size_t)ic * W + xc];
      const float Wn = (rin && xp >= 0 && xp < W) ? DW[o + (size_t)ic * W + xc] : 0.0f;
      const float e = -0.5f * ((DYl + DYn) * oy + (DXl + DXn) * ox);
      const float w = 0.5f * (Wl + Wn);
      float best = 3.4e38f;
      for (int d = -D; d <= D; d++) {
        const int lp = l - d;
        float pv;
        if (lp < 0 || lp >= NL) pv = BIG;
        else pv = rin ? prev[lp] : 0.0f;
        const float q = ((float)d - e) * sct;
        const float pc = dw * (w * fminf(q * q, cap));
        const float v = pv + (lam * pc + P1 * fabsf((float)d));
        best = fminf(best, v);
      }
      sh[l] = best;
      lmin = fminf(lmin, best);
    }
    red[threadIdx.x] = lmin;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) red[threadIdx.x] = fminf(red[threadIdx.x], red[threadIdx.x + s]);
      __syncthreads();
    }
    const float mn = red[0];
    for (int l = threadIdx.x; l < NL; l += blockDim.x) {
      const float v = C[l * HW + (size_t)i * W + x] + sh[l] - mn;
      L[base + l * HW + (size_t)i * W + x] = v; prev[l] = v;
    }
  }
}
"""


# ------------------------------------------------------------------------------------------------ BP
def bp_iterations(U, E, Wt, valid, iters, damp, P1, lam, sct, cap, dw, De):
    """sheetsolve.bp's damped min-sum message passing: U [NL, H, W], valid [4, H, W] (sheetsolve.OFFS order), the
    pairwise cost of edge di computed per step, T(di, d, l) = P1 |d| + lam (dw (w min(((d - e) sct)^2, cap))) with
    e / w = E / Wt [4, NL, H, W] (sheetsolve._dip_ew: the label change the laminations predict toward the neighbour, the
    pair's weight; the PyTorch code's arithmetic). One launch per iteration. Returns U + the sum of the messages."""
    NL, H, W = U.shape
    M = torch.zeros((4, NL, H, W), device=U.device, dtype=torch.float32)
    Mn = torch.empty_like(M)
    nt = _threads(NL)
    cu, ce, cw, cv = _cp(U), _cp(E.float()), _cp(Wt.float()), _cp(valid.float())
    for _ in range(iters):
        _launch("bp_iter", (H * W, 4), (nt,), [cu, _cp(M), _cp(Mn), ce, cw, cv, np.int32(NL), np.int32(H), np.int32(W),
                                               np.int32(De), np.float32(damp), np.float32(1 - damp), np.float32(P1),
                                               np.float32(lam), np.float32(sct), np.float32(cap), np.float32(dw)],
                shared=4 * (NL + nt))
        M, Mn = Mn, M
    return U + (((M[0] + M[1]) + M[2]) + M[3])


def _src_bp():
    return r"""#define BIG 1e4f
extern "C" __global__ void bp_iter(const float* __restrict__ U, const float* __restrict__ M, float* __restrict__ Mn,
    const float* __restrict__ E, const float* __restrict__ Wt, const float* __restrict__ valid, const int NL, const int H,
    const int W, const int De, const float damp, const float omd, const float P1, const float lam, const float sct,
    const float cap, const float dw)
{
  extern __shared__ float sh[];                 // [NL] the neighbour's belief without this edge's message, [blockDim]
  float* red = sh + NL;
  const int cell = blockIdx.x, di = blockIdx.y, i = cell / W, j = cell % W;
  const int oy = di == 0 ? -1 : (di == 1 ? 1 : 0), ox = di == 2 ? -1 : (di == 3 ? 1 : 0), opp = di ^ 1;
  const int ni = i + oy, nj = j + ox;
  const bool in = ni >= 0 && ni < H && nj >= 0 && nj < W;
  const size_t HW = (size_t)H * W, NHW = (size_t)NL * HW;
  for (int l = threadIdx.x; l < NL; l += blockDim.x) {
    float pre = 0.0f;
    if (in) {
      const size_t o = (size_t)l * HW + (size_t)ni * W + nj;
      const float B = U[o] + (((M[o] + M[NHW + o]) + M[2 * NHW + o]) + M[3 * NHW + o]);
      pre = B - M[opp * NHW + o];
    }
    sh[l] = pre;
  }
  __syncthreads();
  const int ND = 2 * De + 1;
  float lmin = 3.4e38f;
  float bl[8]; int nb = 0;                      // (labels of this thread: at most 8 with blockDim >= NL / 8)
  for (int l = threadIdx.x; l < NL; l += blockDim.x) {
    float best = 3.4e38f;
    const size_t oe = (size_t)di * NHW + (size_t)l * HW + (size_t)i * W + j;
    const float e = E[oe], w = Wt[oe];
    for (int d = -De; d <= De; d++) {
      const int lp = l - d;
      const float pv = (lp < 0 || lp >= NL) ? BIG : sh[lp];
      const float q = ((float)d - e) * sct;
      const float v = pv + (P1 * fabsf((float)d) + lam * (dw * (w * fminf(q * q, cap))));
      best = fminf(best, v);
    }
    bl[nb++] = best; lmin = fminf(lmin, best);
  }
  red[threadIdx.x] = lmin;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fminf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mn = red[0], va = valid[(size_t)di * HW + (size_t)i * W + j];
  nb = 0;
  for (int l = threadIdx.x; l < NL; l += blockDim.x) {
    const size_t o = (size_t)di * NHW + (size_t)l * HW + (size_t)i * W + j;
    const float b = (bl[nb++] - mn) * va;
    Mn[o] = damp * M[o] + omd * b;
  }
}
"""


# ------------------------------------------------------------------------------------------------ slope scan
def scan_stack(Xp, Op, ps, lags, gw, D, padk, kst, K, Hc, Wc, UP, axis, step):
    """sheetsolve.lamination_dips_scan_light's semblance stack for one axis ("y": lags along the rows, positions at the
    cell centres along the rows and every px along the columns; "x" the transpose), pooled per cell: Sm, Dm
    [np, Ks, Hc, Wc]. Xp [K + 2 padk, Hp + 2D, Wp + 2D] the padded slab, Op the padded validity. The "x" axis runs the
    "y" kernel on the transposed slab (the same sums)."""
    if axis == "x":
        Sm, Dm = scan_stack(Xp.transpose(1, 2).contiguous(), Op.t().contiguous(), ps, lags, gw, D, padk, kst, K, Wc, Hc, UP, "y", step)
        return Sm.transpose(2, 3).contiguous(), Dm.transpose(2, 3).contiguous()
    npv, nl = len(ps), len(lags)
    Ks = len(range(0, K, kst))
    i0 = np.zeros((npv, nl), np.int32); fr = np.zeros((npv, nl), np.float32)
    for a, p in enumerate(ps):
        for b, d in enumerate(lags):
            f = p * d / step
            i0[a, b] = int(np.floor(f)); fr[a, b] = float(f - int(np.floor(f)))
    dev = Xp.device
    Sm = torch.empty((npv, Ks, Hc, Wc), device=dev, dtype=torch.float32); Dm = torch.empty_like(Sm)
    NC = 8                                                    # cells per block along the columns (NC x UP px staged)
    nkb = -(-Ks // 12); KB = -(-Ks // nkb)                    # depths per block
    imin = int(i0.min()); rows = (KB - 1) * kst + int(i0.max()) - imin + 2
    shared = 4 * max(rows * NC * UP, 2 * KB * NC * npv)       # (the per-cell sums reuse the staged columns' memory)
    args = [_cp(Xp), _cp(Op), _cp(torch.as_tensor(i0, device=dev)), _cp(torch.as_tensor(fr, device=dev)),
            _cp(torch.as_tensor(np.asarray(lags, np.int32), device=dev)), _cp(torch.as_tensor(np.asarray(gw, np.float32), device=dev)),
            _cp(Sm), _cp(Dm), np.int32(Xp.shape[0]), np.int32(Xp.shape[1]), np.int32(Xp.shape[2]), np.int32(nl), np.int32(Ks),
            np.int32(Hc), np.int32(Wc), np.int32(D), np.int32(padk), np.int32(kst), np.int32(UP), np.int32(UP // 2),
            np.int32(KB), np.int32(NC), np.int32(imin), np.int32(rows)]
    _launch("scan_stack", (-(-Wc // NC), Hc, nkb), (KB * NC * UP,), args, shared=shared, key=int(npv))
    return Sm, Dm


def _src_scan(NPV):
    """one thread per (depth, cell, px of the cell's UP): every slope's stack in registers; a block (NC cells of one row
    x KB depths) stages each lag's depth column of its NC x UP px in shared memory once (the slopes' and depths' reads
    overlap there); the UP px of a cell summed in px order through shared memory. Per (slope, cell) the same arithmetic
    in the same order as one thread per cell looping over the px."""
    return r"""
#define NPV %d
extern "C" __global__ void scan_stack(const float* __restrict__ Xp, const float* __restrict__ Op, const int* __restrict__ I0,
    const float* __restrict__ FR, const int* __restrict__ lags, const float* __restrict__ gw, float* __restrict__ Sm,
    float* __restrict__ Dm, const int Kp, const int Hpp, const int Wpp, const int NLAG, const int Ks, const int Hc,
    const int Wc, const int D, const int padk, const int kst, const int UP, const int c, const int KB, const int NC,
    const int imin, const int rows)
{
  extern __shared__ float sh[];
  const int NX = NC * UP;
  float* col = sh;                              // [rows][NX]: one lag's depth column of the block's px
  float* rs = sh;                               // [KB * NC][NPV] x 2: the per-cell sums over the px (after the lags)
  float* rd = rs + KB * NC * NPV;
  const int tid = threadIdx.x;
  const int u = tid %% UP, q = (tid / UP) %% NC, kl = tid / (UP * NC);
  const int jy = blockIdx.y, jx0 = blockIdx.x * NC, k0 = blockIdx.z * KB;
  const int jx = jx0 + q, k = k0 + kl;
  const bool live = jx < Wc && k < Ks;
  const int y0 = c + jy * UP, x0 = jx * UP + u;
  float wsum = 0.0f;
  if (live)
    for (int b = 0; b < NLAG; b++) wsum = wsum + gw[b] * Op[(size_t)(D + lags[b] + y0) * Wpp + D + x0];
  float st[NPV], en[NPV];
#pragma unroll
  for (int p = 0; p < NPV; p++) { st[p] = 0.0f; en[p] = 0.0f; }
  const size_t plane = (size_t)Hpp * Wpp;
  const int kbase = padk + imin + k0 * kst, xbase = D + jx0 * UP, xl = q * UP + u, rk = kl * kst - imin;
  for (int b = 0; b < NLAG; b++) {
    const int yy = D + lags[b] + y0;
    __syncthreads();
    for (int e = tid; e < rows * NX; e += blockDim.x) {
      const int r = e / NX, xx = xbase + e %% NX, ka = kbase + r;
      col[e] = (ka < Kp && xx < Wpp) ? Xp[(size_t)ka * plane + (size_t)yy * Wpp + xx] : 0.0f;
    }
    __syncthreads();
    if (live) {
      const float sq = sqrtf(gw[b] * Op[(size_t)yy * Wpp + D + x0]);
#pragma unroll
      for (int p = 0; p < NPV; p++) {
        const int r = I0[p * NLAG + b] + rk; const float fr = FR[p * NLAG + b];
        const float A = col[r * NX + xl] * sq;
        float ys = A;
        if (fr != 0.0f) {
          const float B = col[(r + 1) * NX + xl] * sq;
          ys = fabsf(fr) < 0.5f ? fmaf(fr, B - A, A) : fmaf(-(B - A), 1.0f - fr, B);
        }
        st[p] = fmaf(ys, sq, st[p]); en[p] = fmaf(ys, ys, en[p]);
      }
    }
  }
  const int cell = kl * NC + q;
  __syncthreads();                              // (the last lag's column read by every thread)
  if (u == 0)
    for (int p = 0; p < NPV; p++) { rs[cell * NPV + p] = 0.0f; rd[cell * NPV + p] = 0.0f; }
  for (int r = 0; r < UP; r++) {                // the px of a cell in order (the sums' rounding as a loop over them)
    __syncthreads();
    if (u == r && live)
      for (int p = 0; p < NPV; p++) {
        rs[cell * NPV + p] = rs[cell * NPV + p] + st[p] * st[p];
        rd[cell * NPV + p] = rd[cell * NPV + p] + en[p] * wsum;
      }
  }
  __syncthreads();
  if (u == 0 && live)
    for (int p = 0; p < NPV; p++) {
      const size_t o = (((size_t)p * Ks + k) * Hc + jy) * Wc + jx;
      Sm[o] = rs[cell * NPV + p] / (float)UP; Dm[o] = rd[cell * NPV + p] / (float)UP;
    }
}
""" % NPV


# ------------------------------------------------------------------------------------------------ streamline median
def streamline_median(St, sy, sk, L):
    """hfib.flow_median3d: per voxel the median of St along its streamline (L px both ways; rows moved by sy, layers by
    sk, clamped; bilinear / trilinear with border clamping as grid_sample(align_corners=True, padding border))."""
    K, Y, X = St.shape
    out = torch.empty_like(St)
    n = K * Y * X
    _launch("streamline_median", (-(-n // 128),), (128,), [_cp(St), _cp(sy), _cp(sk), _cp(out), np.int32(K), np.int32(Y),
                                                          np.int32(X), np.int32(L)])
    return out


def _sortnet(n):
    """Batcher's odd-even merge sort network for n (a power of 2) elements: the compare-exchange pairs."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


def _src_median():
    cs = "\n".join(f"  CS({a},{b})" for a, b in _sortnet(64))
    return r"""
#define CS(a,b) { const float lo = fminf(v[a], v[b]); const float hi = fmaxf(v[a], v[b]); v[a] = lo; v[b] = hi; }
__device__ __forceinline__ float bil(const float* F, float y, float x, const int Y, const int X) {
  y = fminf(fmaxf(y, 0.0f), (float)(Y - 1)); x = fminf(fmaxf(x, 0.0f), (float)(X - 1));
  const int y0 = (int)floorf(y), x0 = (int)floorf(x); const float fy = y - y0, fx = x - x0;
  const int y1 = min(y0 + 1, Y - 1), x1 = min(x0 + 1, X - 1);
  return (1.0f - fy) * ((1.0f - fx) * F[(size_t)y0 * X + x0] + fx * F[(size_t)y0 * X + x1])
       + fy * ((1.0f - fx) * F[(size_t)y1 * X + x0] + fx * F[(size_t)y1 * X + x1]);
}
__device__ __forceinline__ float tril(const float* V, float k, float y, float x, const int K, const int Y, const int X) {
  k = fminf(fmaxf(k, 0.0f), (float)(K - 1));
  const int k0 = (int)floorf(k); const float fk = k - k0; const int k1 = min(k0 + 1, K - 1);
  const size_t S = (size_t)Y * X;
  return (1.0f - fk) * bil(V + k0 * S, y, x, Y, X) + fk * bil(V + k1 * S, y, x, Y, X);
}
extern "C" __global__ void streamline_median(const float* __restrict__ St, const float* __restrict__ fy,
    const float* __restrict__ fk, float* __restrict__ out, const int K, const int Y, const int X, const int L)
{
  const long long n = (long long)K * Y * X;
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  const int x = g % X, y = (g / X) % Y, k = g / ((long long)X * Y);
  float v[64];
  #pragma unroll
  for (int q = 0; q < 64; q++) v[q] = 3.4e38f;
  v[0] = St[g];
  #pragma unroll
  for (int h = 0; h < 2; h++) {
    const float sgn = h == 0 ? 1.0f : -1.0f;
    float yy = (float)y, xx = (float)x, kk = (float)k;
    #pragma unroll
    for (int s = 0; s < 30; s++) {
      if (s < L) {
        const float a = bil(fy, yy, xx, Y, X), b = bil(fk, yy, xx, Y, X);
        yy = yy + sgn * a; kk = fminf(fmaxf(kk + sgn * b, 0.0f), (float)(K - 1)); xx = xx + sgn;
        v[1 + h * 30 + s] = tril(St, kk, yy, xx, K, Y, X);
      }
    }
  }
""" + cs + r"""
  out[g] = v[L];                                  // the lower median of the 2 L + 1 samples (torch.median)
}
"""


# ------------------------------------------------------------------------------------------------ DP paths
def dp_groups(ct, ns, steps, pen):
    """hfib.dp_paths's forward pass for a group of paths: ct [G, n, m] float64 costs, ns [G] the path lengths.
    Returns Dlast [G, m] (float64, the costs at each path's last column) and Pk [G, n, m] (int8, the step taken)."""
    G, n, m = ct.shape
    Dl = torch.empty((G, m), device=ct.device, dtype=torch.float64)
    Pk = torch.full((G, n, m), -1, device=ct.device, dtype=torch.int8)
    nt = _threads(m, 1024)
    _launch("dp_paths", (G,), (nt,), [_cp(ct), _cp(torch.as_tensor(np.asarray(ns, np.int32), device=ct.device)),
                                      _cp(torch.as_tensor(np.asarray(steps, np.int32), device=ct.device)),
                                      _cp(torch.as_tensor(np.asarray(pen, np.float64), device=ct.device)),
                                      _cp(Dl), _cp(Pk), np.int32(n), np.int32(m), np.int32(len(steps))], shared=16 * m)
    return Dl, Pk


def _src_dp():
    return r"""
extern "C" __global__ void dp_paths(const double* __restrict__ ct, const int* __restrict__ ns, const int* __restrict__ steps,
    const double* __restrict__ pen, double* __restrict__ Dl, signed char* __restrict__ Pk, const int n, const int m, const int NS)
{
  extern __shared__ double shd[];               // two rows of m: the current and the next cost-to-come
  double* Da = shd; double* Db = shd + m;
  const int g = blockIdx.x; const double inf = 1e18;
  const double* c = ct + (size_t)g * n * m;
  for (int i = threadIdx.x; i < m; i += blockDim.x) Da[i] = c[i];
  __syncthreads();
  const int last = ns[g] - 1;
  if (last == 0) for (int i = threadIdx.x; i < m; i += blockDim.x) Dl[(size_t)g * m + i] = Da[i];
  for (int t = 1; t < n; t++) {
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
      double best = inf; int si = -1;
      for (int q = 0; q < NS; q++) {
        const int j = i - steps[q];
        const double prev = (j >= 0 && j < m) ? Da[j] : inf;
        const double v = prev + pen[q];
        if (v < best) { best = v; si = q; }
      }
      Db[i] = c[(size_t)t * m + i] + best;
      Pk[((size_t)g * n + t) * m + i] = (signed char)si;
    }
    __syncthreads();
    if (t == last) for (int i = threadIdx.x; i < m; i += blockDim.x) Dl[(size_t)g * m + i] = Db[i];
    double* tmp = Da; Da = Db; Db = tmp;
    __syncthreads();
  }
}
"""



def dp_backtrack(Pk, start, ns, steps):
    """hfib.dp_paths's back-tracking: Pk [G, n, m] int8 (the step taken into each state), start [G] the best end
    states, ns [G] the path lengths -> the states K [G, n] int32 (-1 past a path's end), one thread per path."""
    G, n, m = Pk.shape
    out = torch.full((G, n), -1, device=Pk.device, dtype=torch.int32)
    _launch("dp_backtrack", (-(-G // 128),), (128,), [_cp(Pk), _cp(torch.as_tensor(np.asarray(start, np.int32), device=Pk.device)),
                                                      _cp(torch.as_tensor(np.asarray(ns, np.int32), device=Pk.device)),
                                                      _cp(torch.as_tensor(np.asarray(steps, np.int32), device=Pk.device)), out,
                                                      np.int32(G), np.int32(n), np.int32(m)])
    return out


def _src_dp_back():
    return r"""
extern "C" __global__ void dp_backtrack(const signed char* __restrict__ Pk, const int* __restrict__ start, const int* __restrict__ ns,
    const int* __restrict__ steps, int* __restrict__ out, const int G, const int n, const int m)
{
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  int k = start[g];
  for (int t = ns[g] - 1; t >= 0; t--) {
    out[(size_t)g * n + t] = k;
    const int si = Pk[((size_t)g * n + t) * m + k];
    k = si >= 0 ? k - steps[si] : 0;
  }
}
"""


def follow_dp(Q, lam):
    """hsolve._follow_core's row paths: Q [nF, NL, R2, Xs] the evidence per row offset and column block -> the path [nF, NL,
    Xs] (row offsets): per column the best of the rows -1 / 0 / +1 of the previous column minus lam per row moved (ties:
    the first), back-tracked from the best last row. A block per (fibre, candidate), a thread per row offset."""
    nF, NL, R2, Xs = Q.shape
    B = nF * NL
    path = torch.empty((nF, NL, Xs), device=Q.device, dtype=torch.int32)
    bp = torch.empty((B, Xs, R2), device=Q.device, dtype=torch.int8)
    _launch("follow_dp", (B,), (32,), [_cp(Q.float()), path, bp, np.int32(R2), np.int32(Xs), np.float32(lam)], shared=4 * R2)
    return path.long()


def _src_follow():
    return r"""
extern "C" __global__ void follow_dp(const float* __restrict__ Q, int* __restrict__ path, signed char* __restrict__ bp,
    const int R2, const int Xs, const float lam)
{
  extern __shared__ float sh[];
  const int b = blockIdx.x, r = threadIdx.x;
  const float* q = Q + (size_t)b * R2 * Xs;
  if (r < R2) sh[r] = q[(size_t)r * Xs];
  __syncthreads();
  for (int x = 1; x < Xs; x++) {
    float best = 0.0f; int arg = 0;
    if (r < R2) {                                 // the shifts dy = -1, 0, +1 (rows r + 1, r, r - 1 of the previous column)
      const float c0 = (r + 1 < R2 ? sh[r + 1] : -1e9f) - lam;
      const float c1 = sh[r] - 0.0f;
      const float c2 = (r >= 1 ? sh[r - 1] : -1e9f) - lam;
      best = c0;
      if (c1 > best) { best = c1; arg = 1; }
      if (c2 > best) { best = c2; arg = 2; }
    }
    __syncthreads();
    if (r < R2) { sh[r] = best + q[(size_t)r * Xs + x]; bp[((size_t)b * Xs + x) * R2 + r] = (signed char)arg; }
    __syncthreads();
  }
  if (r == 0) {
    int ix = 0; float mx = sh[0];
    for (int i = 1; i < R2; i++) if (sh[i] > mx) { mx = sh[i]; ix = i; }
    for (int x = Xs - 1; x >= 0; x--) {
      path[(size_t)b * Xs + x] = ix;
      if (x > 0) { const int a = bp[((size_t)b * Xs + x) * R2 + ix]; ix = min(max(ix - (a - 1), 0), R2 - 1); }
    }
  }
}
"""


# ------------------------------------------------------------------------------------------------ separable filters
def sepconv(X, g, axis, mode="replicate", start=0, stride=1, count=None):
    """the correlation of X (contiguous float32) with the 1-D kernel g (odd length, centred) along `axis`, padded by
    replicating (or reflecting) the edges, evaluated at start, start + stride, ... (count positions; default every
    position): Y[.., j, ..] = sum_t g[t] X[.., start + j stride + t - r, ..]. Compiled per kernel length (the tap loop
    unrolled; fused multiply-adds, as cuDNN's convolutions); outputs whose taps lie inside the axis skip the padding."""
    X = X.contiguous().float()
    shp = X.shape
    axis = axis % X.dim()
    outer = int(np.prod(shp[:axis])) if axis else 1
    n = int(shp[axis]); inner = int(np.prod(shp[axis + 1:])) if axis + 1 < X.dim() else 1
    m = int(count if count is not None else n)
    Y = torch.empty(shp[:axis] + (m,) + shp[axis + 1:], device=X.device, dtype=torch.float32)
    tot = outer * m * inner
    if tot == 0:
        return Y
    gk = g.contiguous().float()
    L = int(len(gk))
    if inner > 1 and stride < L and tot < 2 ** 31:            # a first / middle axis: 4 outputs per thread from one window
        nb = -(-m // 4)                                       # of inputs (the threads of a warp read consecutive columns)
        _launch("sepconv_blk", (-(-(outer * nb * inner) // 256),), (256,),
                [X, gk, Y, np.int32(outer), np.int32(n), np.int32(inner), np.int32(start), np.int32(stride), np.int32(m),
                 np.int32(0 if mode == "replicate" else 1), np.int32(L)], opts=("--std=c++14",))
        return Y
    _launch("sepconv", (-(-tot // 256),), (256,), [X, gk, Y, np.int64(outer), np.int32(n), np.int64(inner), np.int32(start),
                                                   np.int32(stride), np.int32(m), np.int32(0 if mode == "replicate" else 1)],
            key=L, opts=("--std=c++14",))
    return Y


def _src_sepconv_blk():
    """sepconv for an axis with inner > 1: a thread computes 4 consecutive outputs from the one window of inputs they
    share ((4 - 1) stride + L values, each read once); per output the same taps in the same order from 0 (the same
    values as one thread per output); one compiled kernel for every tap count."""
    return r"""
#define JB 4
extern "C" __global__ void sepconv_blk(const float* __restrict__ X, const float* __restrict__ g, float* __restrict__ Y,
    const int outer, const int n, const int inner, const int start, const int stride, const int m, const int mode, const int L)
{
  const int nb = (m + JB - 1) / JB;
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= outer * nb * inner) return;
  const int i = gi % inner, rest = gi / inner;
  const int jb = rest % nb, o = rest / nb;
  const float* xp = X + (size_t)o * n * inner + i;
  const int j0 = jb * JB;
  const int c0 = start + j0 * stride - (L - 1) / 2, span = (JB - 1) * stride + L;
  const bool inside = c0 >= 0 && c0 + span - 1 < n;
  float acc[JB];
#pragma unroll
  for (int b = 0; b < JB; b++) acc[b] = 0.0f;
  for (int t = 0; t < span; t++) {
    int q = c0 + t;
    if (!inside) {
      if (mode == 0) q = min(max(q, 0), n - 1);
      else { q = q < 0 ? -q : q; q = q >= n ? 2 * (n - 1) - q : q; }
    }
    const float x = __ldg(xp + (size_t)q * inner);
#pragma unroll
    for (int b = 0; b < JB; b++) {
      const int tt = t - b * stride;
      if (tt >= 0 && tt < L) acc[b] = fmaf(__ldg(g + tt), x, acc[b]);
    }
  }
#pragma unroll
  for (int b = 0; b < JB; b++)
    if (j0 + b < m) Y[((size_t)o * m + j0 + b) * inner + i] = acc[b];
}
"""


def _src_sepconv(L):
    return r"""
#define LT %d
extern "C" __global__ void sepconv(const float* __restrict__ X, const float* __restrict__ g, float* __restrict__ Y,
    const long long outer, const int n, const long long inner, const int start, const int stride, const int m, const int mode)
{
  const long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= outer * m * inner) return;
  const long long i = gi %% inner, rest = gi / inner;
  const int j = (int)(rest %% m); const long long o = rest / m;
  const int c = start + j * stride - (LT - 1) / 2;
  const float* xp = X + o * (long long)n * inner + i;
  float acc = 0.0f;
  if (c >= 0 && c + LT - 1 < n) {
    const float* xc = xp + (long long)c * inner;
#pragma unroll
    for (int t = 0; t < LT; t++) acc = fmaf(__ldg(g + t), __ldg(xc + (long long)t * inner), acc);
  } else {
    for (int t = 0; t < LT; t++) {
      int q = c + t;
      if (mode == 0) q = min(max(q, 0), n - 1);
      else { q = q < 0 ? -q : q; q = q >= n ? 2 * (n - 1) - q : q; }
      acc = fmaf(__ldg(g + t), __ldg(xp + (long long)q * inner), acc);
    }
  }
  Y[gi] = acc;
}
""" % L


# ------------------------------------------------------------------------------------------------ structure tensor
def tensor_dips(comps, okc, f):
    """sheetsolve._dips_from_tensor in one pass: the six tensor components [K, Hc, Wc] -> DY, DX, DW (the dominant
    eigenvector by the same 12 power iterations from the slab normal, on the trace-scaled components)."""
    a = [c_.contiguous().float() for c_ in comps]
    K, Hc, Wc = a[0].shape
    DY = torch.empty_like(a[0]); DX = torch.empty_like(a[0]); DW = torch.empty_like(a[0])
    n = K * Hc * Wc
    _launch("tensor_dips", (-(-n // 256),), (256,), [_cp(x) for x in a] + [_cp(okc.contiguous().float()), _cp(DY), _cp(DX),
                                                                            _cp(DW), np.int64(n), np.int64(Hc * Wc), np.float32(f)])
    return DY, DX, DW


def _src_tensor():
    return r"""
__device__ __forceinline__ float n2n(float x) { return x != x ? 0.0f : (isinf(x) ? copysignf(3.4028234663852886e38f, x) : x); }
extern "C" __global__ void tensor_dips(const float* __restrict__ Jkk, const float* __restrict__ Jyy, const float* __restrict__ Jxx,
    const float* __restrict__ Jky, const float* __restrict__ Jkx, const float* __restrict__ Jyx, const float* __restrict__ okc,
    float* __restrict__ DY, float* __restrict__ DX, float* __restrict__ DW, const long long n, const long long hw, const float f)
{
  const long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= n) return;
  float a = n2n(Jkk[gi]), b = n2n(Jyy[gi]), c = n2n(Jxx[gi]), d = n2n(Jky[gi]), e = n2n(Jkx[gi]), g = n2n(Jyx[gi]);
  const float tr = (a + b) + c;
  const float sc = 1.0f / fmaxf(tr, 1e-30f);
  a = a * sc; b = b * sc; c = c * sc; d = d * sc; e = e * sc; g = g * sc;
  float nk = 1.0f, ny = 0.0f, nx = 0.0f;
  for (int it = 0; it < 12; it++) {
    const float u = (a * nk + d * ny) + e * nx, v = (d * nk + b * ny) + g * nx, w = (e * nk + g * ny) + c * nx;
    const float inv = 1.0f / fmaxf(sqrtf((u * u + v * v) + w * w), 1e-30f);
    nk = u * inv; ny = v * inv; nx = w * inv;
  }
  const float l1 = (nk * ((a * nk + d * ny) + e * nx) + ny * ((d * nk + b * ny) + g * nx)) + nx * ((e * nk + g * ny) + c * nx);
  const float pl = (2.0f * l1 - ((a + b) + c)) / fmaxf(l1, 1e-12f);
  const float s0 = nk + 1e-12f;
  const float sg = s0 > 0.0f ? 1.0f : (s0 < 0.0f ? -1.0f : 0.0f);
  nk = nk * sg; ny = ny * sg; nx = nx * sg;
  const float nkc = fmaxf(nk, 0.3f);
  const float dy = fminf(fmaxf(-ny / nkc, -1.5f), 1.5f), dx = fminf(fmaxf(-nx / nkc, -1.5f), 1.5f);
  DY[gi] = dy * f; DX[gi] = dx * f;
  DW[gi] = fminf(fmaxf(pl, 0.0f), 1.0f) * okc[gi % hw];
}
"""


# ------------------------------------------------------------------------------------------------ Viterbi (fibre tracker)
def viterbi_forward(Sc, lam, ry, rk):
    """hfib._viterbi_fwd_many: Sc [B, Wn, K, Y, Xb] -> F [B, Xb, Wn, K, Y], F[t] = Sc[.., t] + the separable max over the
    (width, layer, row) neighbourhood of F[t - 1] (rows +-ry, layers +-rk, widths +-1; lam per step, applied in the
    same order as hfib._shiftmax: rows, then layers, then widths). One launch per column."""
    Sc = Sc.contiguous().float()
    B, Wn, K, Y, Xb = Sc.shape
    F = torch.empty((B, Xb, Wn, K, Y), device=Sc.device, dtype=torch.float32)
    F[:, 0] = Sc[..., 0]
    n = B * Wn * K * Y
    cs, cf = _cp(Sc), _cp(F)
    for t in range(1, Xb):
        _launch("viterbi_step", (-(-n // 256),), (256,), [cs, cf, np.int32(B), np.int32(Wn), np.int32(K), np.int32(Y), np.int32(Xb),
                                                          np.int32(t), np.int32(ry), np.int32(rk), np.float32(lam[0]),
                                                          np.float32(lam[1]), np.float32(lam[2])])
    return F


def _src_viterbi():
    return r"""
extern "C" __global__ void viterbi_step(const float* __restrict__ Sc, float* __restrict__ F, const int B, const int Wn,
    const int K, const int Y, const int Xb, const int t, const int ry, const int rk, const float lw, const float lk, const float ly)
{
  const long long n = (long long)B * Wn * K * Y;
  const long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= n) return;
  const int y = gi % Y, k = (gi / Y) % K, w = (gi / ((long long)Y * K)) % Wn, b = gi / ((long long)Y * K * Wn);
  const long long S = (long long)Wn * K * Y;
  const float* Fp = F + ((long long)b * Xb + (t - 1)) * S;
  float best = -3.4e38f;
  for (int dw = -1; dw <= 1; dw++) {
    const int w2 = w + dw; if (w2 < 0 || w2 >= Wn) continue;
    for (int dk = -rk; dk <= rk; dk++) {
      const int k2 = k + dk; if (k2 < 0 || k2 >= K) continue;
      for (int dy = -ry; dy <= ry; dy++) {
        const int y2 = y + dy; if (y2 < 0 || y2 >= Y) continue;
        float v = Fp[((long long)w2 * K + k2) * Y + y2];
        for (int q = 0; q < (dy < 0 ? -dy : dy); q++) v = v - ly;
        for (int q = 0; q < (dk < 0 ? -dk : dk); q++) v = v - lk;
        if (dw != 0) v = v - lw;
        best = fmaxf(best, v);
      }
    }
  }
  F[((long long)b * Xb + t) * S + ((long long)w * K + k) * Y + y] = Sc[(((long long)b * Wn + w) * K + k) * Y * Xb + (long long)y * Xb + t] + best;
}
"""


def viterbi_backtrack(F, ends, lam, ry, rk):
    """hfib._backtrack for B volumes at once: F [B, Xb, Wn, K, Y] (the forward scores), ends [B, 3] (w, k, y) -> the
    paths [B, Xb, 3] int (numpy). One thread per volume walks back through the columns: the same candidates in the
    same order (widths, layers, rows), the same float64 arithmetic and the same first-maximum rule."""
    F = F.contiguous().float()
    B, Xb, Wn, K, Y = F.shape
    e = torch.as_tensor(np.asarray(ends, np.int32).reshape(B, 3), device=F.device)
    out = torch.empty((B, Xb, 3), dtype=torch.int32, device=F.device)
    _launch("viterbi_backtrack", (1,), (_threads(B, 1024),), [_cp(F), _cp(e), _cp(out), np.int32(B), np.int32(Xb), np.int32(Wn),
                                                             np.int32(K), np.int32(Y), np.int32(ry), np.int32(rk),
                                                             np.float64(lam[0]), np.float64(lam[1]), np.float64(lam[2])])
    return out.cpu().numpy()


def _src_backtrack():
    return r"""
extern "C" __global__ void viterbi_backtrack(const float* __restrict__ F, const int* __restrict__ ends, int* __restrict__ path,
    const int B, const int Xb, const int Wn, const int K, const int Y, const int ry, const int rk,
    const double l0, const double l1, const double l2)
{
  const int b = threadIdx.x;
  if (b >= B) return;
  const long long S = (long long)Wn * K * Y;
  const float* Fb = F + (long long)b * Xb * S;
  int* pb = path + (long long)b * Xb * 3;
  int w = ends[3 * b], k = ends[3 * b + 1], y = ends[3 * b + 2];
  pb[3 * (Xb - 1)] = w; pb[3 * (Xb - 1) + 1] = k; pb[3 * (Xb - 1) + 2] = y;
  for (int t = Xb - 1; t > 0; t--) {
    bool have = false; double bv = 0.0; int bw = 0, bk = 0, by = 0;
    for (int dw = -1; dw <= 1; dw++)
      for (int dk = -rk; dk <= rk; dk++)
        for (int dy = -ry; dy <= ry; dy++) {
          const int w2 = w + dw, k2 = k + dk, y2 = y + dy;
          if (w2 < 0 || w2 >= Wn || k2 < 0 || k2 >= K || y2 < 0 || y2 >= Y) continue;
          const double v = (((double)Fb[(long long)(t - 1) * S + ((long long)w2 * K + k2) * Y + y2] - l0 * (double)abs(dw))
                            - l1 * (double)abs(dk)) - l2 * (double)abs(dy);
          if (!have || v > bv) { have = true; bv = v; bw = w2; bk = k2; by = y2; }
        }
    w = bw; k = bk; y = by;
    pb[3 * (t - 1)] = w; pb[3 * (t - 1) + 1] = k; pb[3 * (t - 1) + 2] = y;
  }
}
"""


# ------------------------------------------------------------------------------------------------ CT sampling
def slab_sample(vol, table, shape, pz, nz, ks, out, idx):
    """the slab straight from the GPU chunk cache: out[k, idx[i]] = the trilinear CT value at pz[i] + ks[k] nz[i]
    (zyx, float64 points; vol [slots, C, C, C] uint8, table [nz, ny, nx] int32 slots, -1 = absent), the same arithmetic
    as volume.GPUVolume.sample (float64 floor, float32 weights, corners in its order, absent / outside corners 0)."""
    K = len(ks); n = int(pz.shape[0])
    if K * n == 0:
        return out
    C = int(vol.shape[1])
    lc = C.bit_length() - 1
    assert C == 1 << lc and K * n < 2 ** 31 and out.numel() < 2 ** 31
    kt = torch.as_tensor(np.asarray(ks, np.float64), device=pz.device)
    tot = K * n
    _launch("slab_sample", (-(-tot // 256),), (256,), [_cp(vol), _cp(table), _cp(pz.double()), _cp(nz.double()), _cp(kt), _cp(idx.int()), _cp(out),
                                                       np.int32(n), np.int32(K), np.int32(out.shape[1]), np.int32(lc),
                                                       np.int32(table.shape[0]), np.int32(table.shape[1]), np.int32(table.shape[2]),
                                                       np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])])
    return out


def _src_slab():
    return r"""
extern "C" __global__ void slab_sample(const unsigned char* __restrict__ vol, const int* __restrict__ table,
    const double* __restrict__ pz, const double* __restrict__ nz, const double* __restrict__ ks, const int* __restrict__ idx,
    float* __restrict__ out, const int n, const int K, const int npx, const int lc, const int tz, const int ty,
    const int tx, const int sz, const int sy, const int sx)
{
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= n * K) return;
  const int i = gi % n, k = gi / n;
  const int C = 1 << lc, M = C - 1;
  const double kk = ks[k];
  int b[3]; float f[3];
  for (int a = 0; a < 3; a++) {
    const double q = pz[3 * i + a] + kk * nz[3 * i + a];
    const double fl = floor(q);
    b[a] = (int)fl; f[a] = (float)(q - fl);
  }
  const int sh[3] = {sz, sy, sx}; const int tn[3] = {tz, ty, tx};
  float acc = 0.0f;
  bool one = true;                                // all 8 corners inside the volume and in one chunk: one slot lookup
  for (int a = 0; a < 3; a++) one = one && b[a] >= 0 && b[a] + 1 < sh[a] && (b[a] & M) < M;
  if (one) {
    const int slot = table[((b[0] >> lc) * ty + (b[1] >> lc)) * tx + (b[2] >> lc)];
    const size_t base = slot >= 0 ? ((((size_t)slot * C + (b[0] & M)) * C + (b[1] & M)) * C + (b[2] & M)) : 0;
    for (int dz = 0; dz < 2; dz++)
      for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
          const float w = ((dz ? f[0] : 1.0f - f[0]) * (dy ? f[1] : 1.0f - f[1])) * (dx ? f[2] : 1.0f - f[2]);
          const float vv = slot >= 0 ? (float)vol[base + ((size_t)dz * C + dy) * C + dx] : 0.0f;
          acc = acc + w * vv;
        }
  } else {
    for (int dz = 0; dz < 2; dz++)
      for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
          const int c[3] = {b[0] + dz, b[1] + dy, b[2] + dx};
          const float w = ((dz ? f[0] : 1.0f - f[0]) * (dy ? f[1] : 1.0f - f[1])) * (dx ? f[2] : 1.0f - f[2]);
          bool ok = true; int ci[3];
          for (int a = 0; a < 3; a++) {
            ok = ok && c[a] >= 0 && c[a] < sh[a];
            ci[a] = c[a] >> lc;                     // (arithmetic shift: floor division by C)
            ok = ok && ci[a] >= 0 && ci[a] < tn[a];
          }
          float vv = 0.0f;
          if (ok) {
            const int slot = table[(ci[0] * ty + ci[1]) * tx + ci[2]];
            if (slot >= 0) vv = (float)vol[(((size_t)slot * C + (c[0] & M)) * C + (c[1] & M)) * C + (c[2] & M)];
          }
          acc = acc + (ok ? w * vv : 0.0f);
        }
  }
  out[k * npx + idx[i]] = acc;
}
"""


def slab_geometry(Pt, Nt, vt, up):
    """fclib.slab_dense's per-px geometry in one pass: Pt, Nt [H, W, 3] float64 (xyz; 0 at invalid cells), vt [H, W]
    bool -> ok [H up, W up] bool (the 4 bilinear corners and the px's own cell valid), pz, nz [H up W up, 3] float64 (zyx:
    the bilinear position and the unit normal, the PyTorch code's arithmetic in its order)."""
    H, W = int(vt.shape[0]), int(vt.shape[1])
    Ho, Wo = H * up, W * up
    ok = torch.empty((Ho, Wo), dtype=torch.bool, device=Pt.device)
    pz = torch.empty((Ho * Wo, 3), dtype=torch.float64, device=Pt.device); nz = torch.empty_like(pz)
    _launch("slab_geom", (-(-(Ho * Wo) // 256),), (256,), [_cp(Pt.double()), _cp(Nt.double()), _cp(vt.to(torch.uint8)), ok, pz, nz,
                                                         np.int32(H), np.int32(W), np.int32(up), np.float64((up - 1) / (2.0 * up)),
                                                         np.float64(1.0 / up)])
    return ok, pz, nz


def _src_geom():
    return r"""
extern "C" __global__ void slab_geom(const double* __restrict__ P, const double* __restrict__ N, const unsigned char* __restrict__ v,
    bool* __restrict__ ok, double* __restrict__ pz, double* __restrict__ nz, const int H, const int W, const int up, const double off,
    const double inv)
{
  const int Wo = W * up, n = H * up * Wo;
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  const int r = g / Wo, c = g % Wo;
  const double rows = (double)r * inv - off, cols = (double)c * inv - off;   // (x * (1 / up), as PyTorch on CUDA)
  const int i0 = min(max((int)floor(rows), 0), H - 2), j0 = min(max((int)floor(cols), 0), W - 2);
  const double a = fmin(fmax(rows - i0, 0.0), 1.0), b = fmin(fmax(cols - j0, 0.0), 1.0);
  const int ci = min(r / up, H - 1), cj = min(c / up, W - 1);
  ok[g] = v[i0 * W + j0] && v[(i0 + 1) * W + j0] && v[i0 * W + j0 + 1] && v[(i0 + 1) * W + j0 + 1] && v[ci * W + cj];
  const double w00 = (1.0 - a) * (1.0 - b), w10 = a * (1.0 - b), w01 = (1.0 - a) * b, w11 = a * b;
  const size_t q00 = 3 * ((size_t)i0 * W + j0), q10 = 3 * ((size_t)(i0 + 1) * W + j0), q01 = 3 * ((size_t)i0 * W + j0 + 1),
               q11 = 3 * ((size_t)(i0 + 1) * W + j0 + 1);
  double p[3], m[3];
  for (int d = 0; d < 3; d++) {
    p[d] = ((w00 * P[q00 + d] + w10 * P[q10 + d]) + w01 * P[q01 + d]) + w11 * P[q11 + d];
    m[d] = ((w00 * N[q00 + d] + w10 * N[q10 + d]) + w01 * N[q01 + d]) + w11 * N[q11 + d];
  }
  const double nn = fmax(sqrt((m[0] * m[0] + m[1] * m[1]) + m[2] * m[2]), 1e-9);
  for (int d = 0; d < 3; d++) { pz[3 * (size_t)g + d] = p[2 - d]; nz[3 * (size_t)g + d] = m[2 - d] / nn; }
}
"""


def chunk_mark(pz, nz, ks, table, shape, C):
    """the chunks the slab's samples read: flags [tz, ty, tx] uint8, 1 where a corner of the trilinear stencil of a
    sample pz[i] + ks[k] nz[i] lies (slab_sample's own float64 floor; corners outside the volume are read as 0, not
    from a chunk). table [tz, ty, tx] the chunk grid, shape the volume's, C the chunk edge (a power of 2)."""
    K = len(ks); n = int(pz.shape[0])
    flags = torch.zeros(tuple(table.shape), dtype=torch.uint8, device=pz.device)
    if K * n == 0:
        return flags
    assert K * n < 2 ** 31
    kt = torch.as_tensor(np.asarray(ks, np.float64), device=pz.device)
    lc = int(C).bit_length() - 1
    assert int(C) == 1 << lc
    _launch("chunk_mark", (-(-(K * n) // 256),), (256,), [_cp(pz.double()), _cp(nz.double()), _cp(kt), flags, np.int32(n), np.int32(K),
                                                        np.int32(lc), np.int32(table.shape[0]), np.int32(table.shape[1]),
                                                        np.int32(table.shape[2]), np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])])
    return flags


def _src_mark():
    return r"""
extern "C" __global__ void chunk_mark(const double* __restrict__ pz, const double* __restrict__ nz, const double* __restrict__ ks,
    unsigned char* __restrict__ flags, const int n, const int K, const int lc, const int tz, const int ty, const int tx,
    const int sz, const int sy, const int sx)
{
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= n * K) return;
  const int i = gi % n, k = gi / n;
  const double kk = ks[k];
  const int sh[3] = {sz, sy, sx}; const int tn[3] = {tz, ty, tx};
  int c0[3], c1[3];                               // per axis the chunks of the stencil's valid corners (c0 > c1: none)
  for (int a = 0; a < 3; a++) {
    const double q = pz[3 * i + a] + kk * nz[3 * i + a];
    const int b = (int)floor(q);
    int lo = 2, hi = -1;
    for (int d = 0; d < 2; d++) {
      const int c = b + d;
      if (c >= 0 && c < sh[a] && (c >> lc) >= 0 && (c >> lc) < tn[a]) { lo = min(lo, d); hi = max(hi, d); }
    }
    if (hi < lo) return;                          // no corner inside on this axis: the sample reads nothing
    c0[a] = (b + lo) >> lc; c1[a] = (b + hi) >> lc;
  }
  for (int z = c0[0]; z <= c1[0]; z++)
    for (int y = c0[1]; y <= c1[1]; y++)
      for (int x = c0[2]; x <= c1[2]; x++) flags[((size_t)z * ty + y) * tx + x] = 1;
}
"""


# ------------------------------------------------------------------------------------------------ depth shift (shear)
def depth_shift(X, s, valid=False):
    """X [K, H, W] resampled along depth by s [H, W] (slab index units): out[k] = X(k + s) (linear, the index clamped to
    0..K-1, as torch.lerp of the two gathers); with valid also the mask 0 <= k + s <= K - 1. One pass, no index tensors."""
    X = X.contiguous().float()
    K = int(X.shape[0]); HW = int(X[0].numel())
    out = torch.empty_like(X)
    v = torch.empty(X.shape, dtype=torch.bool, device=X.device) if valid else None
    n = K * HW
    assert n < 2 ** 31
    _launch("depth_shift", (-(-n // 256),), (256,), [X, s.contiguous().float(), out, v if valid else np.uint64(0), np.int32(K),
                                                     np.int32(HW), np.int32(1 if valid else 0)])
    return (out, v) if valid else out


def _src_shift():
    return r"""
extern "C" __global__ void depth_shift(const float* __restrict__ X, const float* __restrict__ s, float* __restrict__ out,
    unsigned char* __restrict__ valid, const int K, const int HW, const int want)
{
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= K * HW) return;
  const int p = gi % HW, k = gi / HW;
  float src = (float)k + s[p];
  if (want) valid[gi] = (src >= 0.0f) && (src <= (float)(K - 1));
  src = fminf(fmaxf(src, 0.0f), (float)(K - 1));
  const int i0 = min(max((int)floorf(src), 0), K - 2);
  const float fr = src - (float)i0;
  const float A = X[i0 * HW + p], B = X[(i0 + 1) * HW + p];
  out[gi] = fabsf(fr) < 0.5f ? fmaf(fr, B - A, A) : fmaf(-(B - A), 1.0f - fr, B);
}
"""


# ------------------------------------------------------------------------------------------------ fibre lines
def bar_lines(S, okf, vk, br, bc, sidx, shc, bw, along_rows, minfrac):
    """sheetsolve.bar_descriptors' cross-sections: per bin b a line of 2 bw px through its centre (br, bc) (along the
    rows for the H family, along_rows; else along the columns), read from S [K, Hp, n] (H: column sidx[b]) or
    [K, n, Wp] (V: row sidx[b]); valid where inside, okf > 0.5 and (vk [K, Hp, Wp], or None) set; the line's valid mean
    removed per depth; bins with < minfrac valid px zero; out[k] = the line at depth k - shc[b] (0 outside). Returns
    A [K, nb, 2 bw], okb [nb]. Two passes (the means, then the lines) instead of a dozen over the gathered lines."""
    K = int(S.shape[0]); nb = int(br.numel()); L2 = 2 * int(bw)
    Hp, Wp = int(okf.shape[0]), int(okf.shape[1])
    A = torch.empty((K, nb, L2), device=S.device, dtype=torch.float32)
    okb = torch.empty(nb, device=S.device, dtype=torch.bool)
    if K * nb == 0:
        return A, okb
    mean = torch.empty((K, nb), device=S.device, dtype=torch.float32)
    i32 = lambda x: _cp(x.to(torch.int32))
    args = [_cp(S.float()), _cp(okf.float()), _cp(vk) if vk is not None else np.uint64(0), i32(br), i32(bc), i32(sidx), i32(shc),
            mean, okb, A, np.int32(K), np.int32(nb), np.int32(L2), np.int32(Hp), np.int32(Wp), np.int32(S.shape[1]), np.int32(S.shape[2]),
            np.int32(1 if along_rows else 0), np.int32(1 if vk is not None else 0), np.float32(minfrac)]
    _launch("bar_lines", (-(-(K * nb) // 128),), (128,), args + [np.int32(0)])
    _launch("bar_lines", (-(-(K * nb * L2) // 256),), (256,), args + [np.int32(1)])
    return A, okb


def _src_bar_lines():
    return r"""
__device__ __forceinline__ bool px_of(const int b, const int j, const int* br, const int* bc, const int bw, const int rowsdir,
    const int Hp, const int Wp, int& r, int& c)
{
  r = rowsdir ? br[b] + j - bw : br[b]; c = rowsdir ? bc[b] : bc[b] + j - bw;
  const bool ins = r >= 0 && r < Hp && c >= 0 && c < Wp;
  r = min(max(r, 0), Hp - 1); c = min(max(c, 0), Wp - 1);
  return ins;
}
extern "C" __global__ void bar_lines(const float* __restrict__ S, const float* __restrict__ okf, const unsigned char* __restrict__ vk,
    const int* __restrict__ br, const int* __restrict__ bc, const int* __restrict__ sidx, const int* __restrict__ shc,
    float* __restrict__ mean, bool* __restrict__ okb, float* __restrict__ A, const int K, const int nb, const int L2, const int Hp,
    const int Wp, const int S1, const int S2, const int rowsdir, const int hasvk, const float minfrac, const int phase)
{
  const int bw = L2 / 2;
  if (phase == 0) {                               // per (depth, bin): the valid px's mean; per bin (depth 0): okb
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= K * nb) return;
    const int b = gi % nb, k = gi / nb;
    float s = 0.0f; int cnt = 0, cpx = 0;
    for (int j = 0; j < L2; j++) {
      int r, c;
      const bool ins = px_of(b, j, br, bc, bw, rowsdir, Hp, Wp, r, c);
      const bool mk = ins && okf[(size_t)r * Wp + c] > 0.5f;
      cpx += mk;
      if (mk && (!hasvk || vk[((size_t)k * Hp + r) * Wp + c])) {
        s += rowsdir ? S[((size_t)k * S1 + r) * S2 + sidx[b]] : S[((size_t)k * S1 + sidx[b]) * S2 + c];
        cnt++;
      }
    }
    mean[gi] = s / (float)max(cnt, 1);
    if (k == 0) okb[b] = (float)cpx / (float)L2 >= minfrac;
    return;
  }
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;   // per (depth, bin, px): the line at the frame depth
  if (gi >= K * nb * L2) return;
  const int j = gi % L2, b = (gi / L2) % nb, k = gi / (L2 * nb);
  const int k0 = k - shc[b];
  float out = 0.0f;
  if (k0 >= 0 && k0 < K && okb[b]) {
    int r, c;
    const bool ins = px_of(b, j, br, bc, bw, rowsdir, Hp, Wp, r, c);
    const float m = (ins && okf[(size_t)r * Wp + c] > 0.5f && (!hasvk || vk[((size_t)k0 * Hp + r) * Wp + c])) ? 1.0f : 0.0f;
    const float v = rowsdir ? S[((size_t)k0 * S1 + r) * S2 + sidx[b]] : S[((size_t)k0 * S1 + sidx[b]) * S2 + c];
    out = (v * m - mean[(size_t)k0 * nb + b]) * m;
  }
  A[gi] = out;
}
"""


# ------------------------------------------------------------------------------------------------ fibre tie products
def shift_dot(A, An, lag, gs, D):
    """sheetsolve.bar_cost's products for the across-fibre lags gs = (-lag, 0, lag) and every depth shift d in -D..D at
    once: num[g, d, k, h, w] = sum_j A[k, lag + j, h, w] An[(k - d) mod K, lag + g + j, h, w], j < n - 2 lag (A, An
    [K, n, hb, wb]; the wrapped depths as the roll / modulo of the PyTorch code, masked by the caller). One pass over the
    lines; each thread keeps the 2 lag + 1 values of An the three lags read in a shift register (each value loaded once)."""
    assert tuple(int(g) for g in gs) == (-int(lag), 0, int(lag))
    A = A.contiguous().float(); An = An.contiguous().float()
    K, n, hb, wb = A.shape
    ND = 2 * D + 1
    out = torch.empty((3, ND, K, hb, wb), device=A.device, dtype=torch.float32)
    tot = K * hb * wb
    _launch("shift_dot", (-(-tot // 128),), (128,), [A, An, out, np.int32(K), np.int32(n), np.int32(hb * wb), np.int32(D)],
            key=(ND, int(lag)), opts=("--std=c++14",))
    return out


def _src_shift_dot(key):
    ND, LG = key
    return r"""
#define NDN %d
#define LG %d
#define WN (2 * LG + 1)
extern "C" __global__ void shift_dot(const float* __restrict__ A, const float* __restrict__ An, float* __restrict__ out,
    const int K, const int n, const int HW, const int D)
{
  const long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= (long long)K * HW) return;
  const int p = (int)(gi %% HW), k = (int)(gi / HW);
  float acc[3][NDN], win[NDN][WN];
  long long kk[NDN];
#pragma unroll
  for (int d = 0; d < NDN; d++) {
    kk[d] = (long long)(((k - (d - D)) %% K + K) %% K) * n;
#pragma unroll
    for (int g = 0; g < 3; g++) acc[g][d] = 0.0f;
#pragma unroll
    for (int q = 0; q < WN - 1; q++) win[d][q + 1] = __ldg(An + (kk[d] + q) * HW + p);
  }
  const int m = n - 2 * LG;
  for (int j = 0; j < m; j++) {                 // win[d][q] = An[(k - d) mod K, j + q]: the lags -LG, 0, LG read q = 0, LG, 2 LG
    const float a = __ldg(A + ((long long)k * n + LG + j) * HW + p);
#pragma unroll
    for (int d = 0; d < NDN; d++) {
#pragma unroll
      for (int q = 0; q < WN - 1; q++) win[d][q] = win[d][q + 1];
      win[d][WN - 1] = __ldg(An + (kk[d] + j + 2 * LG) * HW + p);
    }
#pragma unroll
    for (int g = 0; g < 3; g++)
#pragma unroll
      for (int d = 0; d < NDN; d++) acc[g][d] = fmaf(a, win[d][g * LG], acc[g][d]);
  }
#pragma unroll
  for (int g = 0; g < 3; g++)
#pragma unroll
    for (int d = 0; d < NDN; d++) out[(((long long)g * NDN + d) * K + k) * HW + p] = acc[g][d];
}
""" % (ND, LG)

SOURCES = {"sgm_dip": _src_sgm, "bp_iter": _src_bp, "scan_stack": _src_scan, "streamline_median": _src_median,
           "dp_paths": _src_dp, "dp_backtrack": _src_dp_back, "follow_dp": _src_follow, "sepconv": _src_sepconv, "sepconv_blk": _src_sepconv_blk, "tensor_dips": _src_tensor, "viterbi_step": _src_viterbi,
           "viterbi_backtrack": _src_backtrack, "slab_sample": _src_slab, "slab_geom": _src_geom, "chunk_mark": _src_mark, "depth_shift": _src_shift, "shift_dot": _src_shift_dot,
           "bar_lines": _src_bar_lines}

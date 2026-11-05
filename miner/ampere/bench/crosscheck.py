"""Cross-hardware determinism check for the integer runtime (design doc, build step 2).

Identical inputs (from NumPy seeds) on every machine; outputs are saved so runs on
different GPUs (and CPU) can be compared byte for byte with compare_crosscheck.py.

  L1 mining GEMM      int32 output of MinedWeight.forward + winning tiles (consensus-critical)
  L2 quantizer        fused Hadamard + per-token quantizer: int8 values and fp32 scales
  L3 full model       Qwen3 mined forward: sha256 of every decoder layer's output
  L4 prefill/decode   same GPU: last-token hidden state from full prefill vs KV-cache decode
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from uwc_ampere import model as M
from uwc_ampere import ops
from uwc_ampere import pearl_glue as pg


def sha(x) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().contiguous().cpu()
        x = x.view(torch.int16).numpy() if x.dtype == torch.bfloat16 else x.numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:20]


def bf16_from_np(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)


def quant_cpu_emulation(x: np.ndarray, bs: int):
    """Same float32 op order as quant_rows_had (IEEE adds, max, divide)."""
    v = x.astype(np.float32).copy()
    m, k = v.shape
    if bs > 1:
        h = 1
        while h < bs:
            v = v.reshape(m, k // (2 * h), 2, h)
            a, b = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
            v[:, :, 0, :], v[:, :, 1, :] = a + b, a - b
            v = v.reshape(m, k)
            h *= 2
        norm = np.float32(1.0) / np.sqrt(np.float32(bs))
    else:
        norm = np.float32(1.0)
    mx = np.maximum(np.abs(v).max(axis=1), np.float32(1e-8))
    sc = (mx / np.float32(64.0)).astype(np.float32)
    q = np.clip(np.rint(v / sc[:, None]), -64, 64).astype(np.int8)
    return q, (sc * norm).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/Documents/useful-work-coin/models/Qwen3-1.7B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true", help="CPU references for L1/L2 only")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    res = {"device": "cpu" if a.cpu else torch.cuda.get_device_name(), "torch": torch.__version__}
    arrays = {}

    # ---- L1: mining GEMM
    m, n, k = 1024, 3072, 4096
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    nbits = 0x1E00FFFF
    hdr = pg.header(nbits, 99)
    cfg = pg.mining_config(k)
    key = pg.job_key(hdr, cfg)
    if a.cpu:
        C = (A.astype(np.int64) @ Bt.astype(np.int64).T).astype(np.int32)
        res["L1_gemm_int32"] = sha(C)
    else:
        w = ops.MinedWeight(Bt=torch.from_numpy(Bt).cuda(), b_scale=torch.ones(n, device="cuda"), max_wins=4096)
        w.setup_block(key, pg.bound_words(pg.jackpot_bound(nbits, cfg)))
        C = w.forward(torch.from_numpy(A).cuda(), torch.ones(m, device="cuda"), out_mode=ops.OUT_I32)
        res["L1_gemm_int32"] = sha(C)
        wins = sorted(w.winners(m))
        res["L1_winners"] = len(wins)
        res["L1_winners_sha"] = hashlib.sha256(repr(wins).encode()).hexdigest()[:20]

    # ---- L2: quantizer
    x = (rng.standard_normal((1024, 4096)).astype(np.float32) * np.linspace(0.05, 6, 4096, dtype=np.float32))
    x[:, 7] *= 40.0  # an outlier channel, as in real activations
    xb = bf16_from_np(x)
    for bs in (0, 128, 512):
        if a.cpu:
            q, s = quant_cpu_emulation(xb.float().numpy(), bs)
        else:
            qt, st = ops.quant_rows_had(xb.cuda().contiguous(), bs)
            q, s = qt.cpu().numpy(), st.cpu().numpy()
        res[f"L2_q_bs{bs}"], res[f"L2_scale_bs{bs}"] = sha(q), sha(s)
        arrays[f"L2_q_bs{bs}"], arrays[f"L2_scale_bs{bs}"] = q, s

    if not a.cpu:
        from transformers import AutoModelForCausalLM
        mdl = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
        mls = M.convert_qwen3(mdl, "cuda", mode="mined", hadamard_block=512)
        mdl.to("cuda")
        for ml in mls:
            ml.setup_block(lambda kk: pg.job_key(hdr, pg.mining_config(kk)),
                           lambda kk: pg.bound_words(pg.jackpot_bound(0x1D00FFFF, pg.mining_config(kk))))
        ids = torch.from_numpy(rng.integers(0, 150000, size=(2, 512))).cuda()

        # ---- L3: per-layer hashes of a full mined forward
        layer_sha = []
        hooks = [lay.register_forward_hook(lambda mod, inp, out: layer_sha.append(
            sha(out[0] if isinstance(out, tuple) else out))) for lay in mdl.model.layers]
        qkv0 = mls[0]
        cap = {}

        def grab(mod, inp, out):
            cap["in"], cap["out"] = inp[0].detach().clone(), out.detach().clone()
        hq = qkv0.register_forward_hook(grab)
        with torch.no_grad():
            h = mdl.model(input_ids=ids).last_hidden_state
        for hk in hooks + [hq]:
            hk.remove()
        res["L3_layer_sha"] = layer_sha
        res["L3_final_sha"] = sha(h)
        res["L3_layer0_qkv_in_sha"], res["L3_layer0_qkv_out_sha"] = sha(cap["in"]), sha(cap["out"])
        arrays["L3_final"] = h.float().cpu().numpy()
        arrays["L3_layer0_qkv_in"] = cap["in"].float().cpu().numpy()

        # ---- L4: prefill vs KV-cache decode on this GPU
        with torch.no_grad():
            full = mdl.model(input_ids=ids).last_hidden_state[:, -1]
            pre = mdl.model(input_ids=ids[:, :-1], use_cache=True)
            dec = mdl.model(input_ids=ids[:, -1:], past_key_values=pre.past_key_values, use_cache=True).last_hidden_state[:, -1]
        res["L4_prefill_vs_decode_identical"] = bool(torch.equal(full, dec))
        res["L4_frac_elements_differ"] = float((full != dec).float().mean().item())
        res["L4_max_abs_diff"] = float((full.float() - dec.float()).abs().max().item())

    json.dump(res, open(a.out + ".json", "w"), indent=1)
    np.savez_compressed(a.out + ".npz", **arrays)
    print(json.dumps({kk: v for kk, v in res.items() if kk != "L3_layer_sha"}, indent=1))


if __name__ == "__main__":
    main()

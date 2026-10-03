"""Mean eval return over the last 10 evals, best and final, for every run log in runs/ablations/logs (pass that folder as the first argument).
SAC logs evaluate every 5k steps, VPG logs every 20 epochs (82k steps) or 5 epochs (82k steps for the 16k batch)."""
import re, glob, pathlib, sys
S = pathlib.Path(sys.argv[1])
runs = {
    "savings, 100k (reference)": ["run_savings.log", "run_savings_s1.log"],
    "mix 20% solar-only": ["run_mix_s0.log", "run_mix_s1.log"],
    "short 48h episodes": ["run_short_s0.log"],
    "residual action": ["run_res_s0.log", "run_res_s1.log"],
    "S2 no forecasts": ["run_s2_s0.log", "run_s2_s1.log"],
    "shaped SoC 0.20": ["run_shaped_s0.log", "run_shaped_s1.log"],
    "n-step 4": ["n4_s0.log", "n4_s1.log"],
    "n-step 8": ["n8_s0.log", "n8_s1.log"],
    "batch 512, lr 1e-4": ["b512_s0.log", "b512_s1.log"],
    "plain, 400k": ["plain400k_s0.log", "plain400k_s1.log"],
    "batch 512, lr 1e-4, 400k": ["b512_400k_s0.log"],
    "SAC + clip penalty 0.05": ["sac_clip_s0.log", "sac_clip_s1.log"],
    "SAC + clip penalty, 200k": ["sac_clip_200k_s2.log"],
    "VPG plain, pi lr 3e-4 (stopped)": ["vpg_plain_lr3e-4_s0.log", "vpg_plain_lr3e-4_s1.log"],
    "VPG plain, pi lr 1e-3 (stopped)": ["vpg_plain_lr1e-3_s0.log"],
    "VPG residual (stopped)": ["vpg_residual_s0.log", "vpg_residual_s1.log"],
    "VPG residual + shaping (stopped)": ["vpg_residual_shaped_s0.log", "vpg_residual_shaped_s1.log"],
    "VPG residual, 16k batch (stopped)": ["vpg_residual_batch16k_s0.log"],
    "VPG residual + clip penalty": ["vpg_residual_clip_s0.log", "vpg_residual_clip_s1.log"],
    "iface power + clip (ref)": ["sac_clip_s0.log", "sac_clip_s1.log"],
    "iface target_soc": ["tsoc_s0.log", "tsoc_s1.log", "tsoc_s2.log"],
    "iface target_soc + clip": ["tsoc_clip_s0.log", "tsoc_clip_s1.log", "tsoc_clip_s2.log"],
    "iface residual (env) + clip": ["res_clip_s0.log", "res_clip_s1.log", "res_clip_s2.log"],
    # from here on: train split only, eval = mean over the 26 validation weeks (not comparable with the rows above)
    "SPLIT target_soc": ["tsoc_split_s0.log", "tsoc_split_s1.log", "tsoc_split_s2.log"],
    "SPLIT power + clip": ["pclip_split_s0.log", "pclip_split_s1.log"],
}
print(f"{'variant':34s} {'seed':>4s} {'last-10 evals':>14s} {'best':>6s} {'final':>6s} {'evals':>6s}")
for name, files in runs.items():
    for i, f in enumerate(files):
        p = S / f
        if not p.exists(): print(f"{name:28s} {i:>4d}  (missing)"); continue
        ev = [float(r) for r in re.findall(r"(?:step|steps)\s+\d+ eval return\s+([-\d.]+)", p.read_text())]
        tail = ev[-10:]
        print(f"{name:34s} {i:>4d} {sum(tail)/len(tail):14.2f} {max(ev):6.2f} {ev[-1]:6.2f} {len(ev):5d}")

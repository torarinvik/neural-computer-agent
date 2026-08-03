"""Does a deeper ancestry reach the next slot's input at all?

A new slot reads cat([state.hidden, event]). The shared base is frozen, so the
only route from an earlier rung to that input is behavioral: an earlier slot
changes the logits, which changes the action, which feeds the recurrent state. A
rectified gate that is exactly shut on a foreign task's events changes none of
them, and then every deeper parent hands the next slot bit-identical numbers.

Run this before assuming a rung can inherit anything. A distance of zero here
means no experiment on that pair can show transfer, however many seeds it uses.
"""
import torch
from experiments.archive.unified_cognitive_controller.legacy_model import UnifiedCognitiveController
from experiments.archive.unified_cognitive_controller.environment import generate_lifetimes

CK = {
 "1skill_6810": "artifacts/checkpoints/unified_memory_online_utility_seed6810.pt",
 "2skill_8397": "artifacts/checkpoints/unified_binary_context_integrated_seed8397.pt",
 "3skill_8413": "artifacts/checkpoints/unified_three_skill_compounding_seed8413.pt",
 "4skill_8600": "artifacts/checkpoints/depth/depth_rung4_8600.pt",
 "5skill_8600": "artifacts/checkpoints/depth/depth_rung5_8600.pt",
}
models, states = {}, {}
for n, p in CK.items():
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = UnifiedCognitiveController(**d["model_configuration"]); m.load_state_dict(d["state_dict"]); m.eval()
    models[n] = m; states[n] = d["state_dict"]

print("=== is the shared base identical across ancestry? ===")
names = list(CK)
base_keys = [k for k in states["1skill_6810"] if "adapter" not in k]
for a, b in zip(names, names[1:]):
    same = sum(torch.equal(states[a][k], states[b][k]) for k in base_keys if k in states[b])
    total = sum(1 for k in base_keys if k in states[b])
    print(f"  {a:12s} -> {b:12s}  base tensors identical: {same}/{total}")

print()
print("=== features the NEXT slot would read: cat([state.hidden, event]) ===")
batch = generate_lifetimes(128, 6, seed=7777, task="context_identity_and", support_trials=2)
feats = {}
with torch.no_grad():
    for n, m in models.items():
        st = m.initial_state(128, device="cpu")
        act = torch.full((128,), 2, dtype=torch.long); rew = torch.zeros(128)
        collected = []
        for t in range(4):
            ev = m.vision(batch.frames[:, t])
            collected.append(torch.cat([st.hidden, ev], dim=-1))
            out, st = m.step(batch.frames[:, t], st, act, rew, torch.zeros(128))
            act = out.logits.argmax(-1)
            rew = (act == batch.correct_actions[:, t]).float()
        feats[n] = torch.cat(collected)
ref = feats["3skill_8413"]
scale = ref.std()
for n in names:
    d = (feats[n] - ref).norm(dim=-1).mean() / scale
    print(f"  {n:12s} distance from 3-skill features: {d:.6f}")

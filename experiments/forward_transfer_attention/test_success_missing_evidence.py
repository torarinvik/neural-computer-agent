import torch

from .audit_success_missing_evidence import ablate_policy_frames


def test_pixel_ablation_preserves_only_requested_frames() -> None:
    frames = torch.arange(
        2 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 5, 3, 2, 2)
    no_feedback = ablate_policy_frames(frames, "no_feedback")
    assert torch.equal(no_feedback[:, :2], frames[:, :2])
    assert bool((no_feedback[:, 2] == 0).all())
    assert torch.equal(no_feedback[:, 3:], frames[:, 3:])

    feedback_only = ablate_policy_frames(frames, "feedback_only")
    assert bool((feedback_only[:, :2] == 0).all())
    assert torch.equal(feedback_only[:, 2], frames[:, 2])
    assert bool((feedback_only[:, 3:] == 0).all())

    order_only = ablate_policy_frames(frames, "order_only")
    assert torch.equal(order_only[:, :2], frames[:, :2])
    assert bool((order_only[:, 2:] == 0).all())


def test_normal_ablation_is_an_independent_copy() -> None:
    frames = torch.randn(3, 5, 3, 4, 4)
    normal = ablate_policy_frames(frames, "normal")
    assert torch.equal(normal, frames)
    assert normal.data_ptr() != frames.data_ptr()

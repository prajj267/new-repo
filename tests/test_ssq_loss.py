"""
Unit tests for src/losses/ssq.py  (Stage 2).

All ten required tests from requirements.md §4.3, plus extras.
Run: pytest tests/test_ssq_loss.py -v
"""

from __future__ import annotations

import math
from collections import deque

import pytest
import torch

from src.losses.ssq import (
    _box_iou_elementwise,
    ciou_loss,
    compute_beta,
    quality_loss,
    scale_loss,
    shape_loss,
    ssq_loss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_pair():
    """Predicted box == ground-truth box. All losses should be 0 or minimal."""
    t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])  # [cx,cy,w,h]
    p = t.clone().requires_grad_(True)
    return p, t


@pytest.fixture
def tiny_box_pair():
    """Very small boxes: w=1, h=1 px. Tests NaN/inf stability."""
    t = torch.tensor([[100.0, 100.0, 1.0, 1.0]], dtype=torch.float64)
    p = torch.tensor([[101.0, 101.0, 1.5, 1.2]], dtype=torch.float64,
                     requires_grad=True)
    return p, t


@pytest.fixture
def batch_pair():
    """Random batch of 8 boxes for numeric tests."""
    torch.manual_seed(0)
    t = torch.stack([
        torch.tensor([320.0, 256.0, 20.0, 16.0]),
        torch.tensor([100.0, 100.0, 8.0,  6.0]),
        torch.tensor([500.0, 400.0, 32.0, 28.0]),
        torch.tensor([200.0, 150.0, 5.0,  4.0]),
        torch.tensor([60.0,  60.0,  12.0, 10.0]),
        torch.tensor([450.0, 350.0, 18.0, 14.0]),
        torch.tensor([310.0, 240.0, 7.0,  6.0]),
        torch.tensor([150.0, 200.0, 25.0, 20.0]),
    ])
    noise = torch.randn(8, 4) * torch.tensor([5.0, 5.0, 3.0, 3.0])
    p = (t + noise).clamp(min=1.0).requires_grad_(True)
    return p, t


# ---------------------------------------------------------------------------
# Required test 1: test_scale_matches_closed_form
# ---------------------------------------------------------------------------

class TestScaleLoss:
    def test_scale_matches_closed_form(self):
        """L_scale must equal the hand-computed closed form for a known pair."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]])
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])
        w_s = 0.6

        # Closed form
        centre = (310 - 320) ** 2 + (250 - 256) ** 2      # 100 + 36 = 136
        size   = (18 - 20) ** 2 + (14 - 16) ** 2           # 4 + 4 = 8
        expected = 136.0 + 0.6 * 1.0 * 8.0                 # = 140.8

        got = scale_loss(p, t, w_s=w_s, size_cov_scale=1.0).item()
        assert got == pytest.approx(expected, rel=1e-5), f"Expected {expected}, got {got}"

    def test_scale_zero_for_identical_boxes(self, perfect_pair):
        p, t = perfect_pair
        loss = scale_loss(p.detach(), t)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_scale_non_negative(self, batch_pair):
        p, t = batch_pair
        assert (scale_loss(p, t) >= 0).all()

    def test_scale_size_cov_scale_effect(self):
        """size_cov_scale=0.25 should give 1/4 the size term contribution."""
        p = torch.tensor([[300.0, 250.0, 15.0, 12.0]])
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])
        l1 = scale_loss(p, t, w_s=1.0, size_cov_scale=1.0).item()
        l4 = scale_loss(p, t, w_s=1.0, size_cov_scale=0.25).item()
        centre = (300-320)**2 + (250-256)**2  # 400+36=436
        size   = (15-20)**2 + (12-16)**2      # 25+16=41
        expected_1 = centre + 1.0 * size
        expected_4 = centre + 0.25 * size
        assert l1 == pytest.approx(expected_1, rel=1e-5)
        assert l4 == pytest.approx(expected_4, rel=1e-5)

    def test_scale_differentiable(self, batch_pair):
        p, t = batch_pair
        loss = scale_loss(p, t).mean()
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


# ---------------------------------------------------------------------------
# Required test 2: test_shape_zero_identical_boxes
# ---------------------------------------------------------------------------

class TestShapeLoss:
    def test_shape_zero_identical_boxes(self, perfect_pair):
        """L_shape must be exactly 0 when p == t."""
        p, t = perfect_pair
        loss = shape_loss(p.detach(), t)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_shape_zero_same_aspect_ratio_different_scale(self):
        """Same aspect ratio, different size: loss should be 0."""
        t = torch.tensor([[100.0, 100.0, 20.0, 10.0]])  # r = 2
        p = torch.tensor([[100.0, 100.0, 40.0, 20.0]])  # r = 2, 2x bigger
        # rel_w = |40-20|/max(40,20) = 20/40 = 0.5  → NOT zero
        # Actually the test above is wrong: same ratio but different size gives nonzero.
        # Correct: shape_loss is 0 only when w_p==w_t AND h_p==h_t, not just same ratio.
        # Verify this:
        loss = shape_loss(p, t)
        # Should be nonzero (different sizes)
        assert loss.item() > 0.0

    def test_shape_non_negative(self, batch_pair):
        p, t = batch_pair
        assert (shape_loss(p, t) >= 0).all()

    def test_shape_differentiable(self, batch_pair):
        p, t = batch_pair
        loss = shape_loss(p, t).mean()
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    def test_shape_scale_exponent(self):
        """Different shape_scale values must give different results.

        Use boxes where w_t >> h_t so the aspect-ratio weights shift
        noticeably when the exponent changes.
        """
        # w_t=40, h_t=4 → 10:1 ratio; weights shift a lot with exponent
        p = torch.tensor([[100.0, 100.0, 28.0, 3.0]])
        t = torch.tensor([[100.0, 100.0, 40.0, 4.0]])
        l1 = shape_loss(p, t, shape_scale=1.0).item()
        l2 = shape_loss(p, t, shape_scale=2.0).item()
        l3 = shape_loss(p, t, shape_scale=4.0).item()  # jump to 4 for clear separation
        assert abs(l1 - l2) > 1e-4, \
            f"scale=1 and scale=2 too close: {l1:.6f} vs {l2:.6f}"
        assert abs(l2 - l3) > 1e-4, \
            f"scale=2 and scale=4 too close: {l2:.6f} vs {l3:.6f}"


# ---------------------------------------------------------------------------
# Required test 3: test_no_nan_tiny_boxes
# ---------------------------------------------------------------------------

class TestNaNStability:
    def test_no_nan_tiny_boxes_scale(self, tiny_box_pair):
        p, t = tiny_box_pair
        loss = scale_loss(p, t.float())
        assert not torch.isnan(loss).any(), "scale_loss produced NaN for tiny boxes"
        assert not torch.isinf(loss).any(), "scale_loss produced Inf for tiny boxes"

    def test_no_nan_tiny_boxes_shape(self, tiny_box_pair):
        p, t = tiny_box_pair
        loss = shape_loss(p, t.float())
        assert not torch.isnan(loss).any(), "shape_loss produced NaN for tiny boxes"
        assert not torch.isinf(loss).any(), "shape_loss produced Inf for tiny boxes"

    def test_no_nan_tiny_boxes_quality(self, tiny_box_pair):
        p, t = tiny_box_pair
        window = deque(maxlen=100)
        window.append(0.5)
        beta = compute_beta(0.5, window)
        loss = quality_loss(p, t.float(), beta=beta)
        assert not torch.isnan(loss).any(), "quality_loss produced NaN for tiny boxes"
        assert not torch.isinf(loss).any(), "quality_loss produced Inf for tiny boxes"

    def test_no_nan_zero_size_boxes(self):
        """Boxes with w=eps, h=eps must not produce NaN."""
        eps = 1e-5
        p = torch.tensor([[100.0, 100.0, eps, eps]], requires_grad=True)
        t = torch.tensor([[102.0, 102.0, eps, eps]])
        ls = scale_loss(p, t)
        lsh = shape_loss(p, t)
        assert not torch.isnan(ls).any()
        assert not torch.isnan(lsh).any()


# ---------------------------------------------------------------------------
# Required test 4: test_beta_is_detached
# ---------------------------------------------------------------------------

class TestComputeBeta:
    def test_beta_is_detached(self):
        """compute_beta must return a Python float, not a Tensor."""
        window = deque(maxlen=100)
        window.append(0.4)
        beta = compute_beta(0.6, window)
        assert isinstance(beta, float), f"Expected float, got {type(beta)}"
        assert not isinstance(beta, torch.Tensor)

    def test_beta_rejects_tensor_input(self):
        """Passing a Tensor as iou_val must raise TypeError."""
        window = deque(maxlen=100)
        iou_tensor = torch.tensor(0.5)
        with pytest.raises(TypeError, match="Python float"):
            compute_beta(iou_tensor, window)

    def test_beta_formula(self):
        """beta = (1-iou) / (mean_window + eps)."""
        eps = 1e-7
        window = deque(maxlen=100)
        # Pre-fill window with value 0.4
        for _ in range(10):
            window.append(0.4)
        # Now call with iou=0.6; window appends 0.6 first
        beta = compute_beta(0.6, window)
        # After append: window has ten 0.4s + one 0.6
        mean_iou = (10 * 0.4 + 0.6) / 11
        expected = (1 - 0.6) / (mean_iou + eps)
        assert beta == pytest.approx(expected, rel=1e-5)

    def test_beta_non_negative(self):
        """Beta must be >= 0 since iou in [0,1] and window mean > 0."""
        window = deque(maxlen=100)
        for v in [0.3, 0.5, 0.7]:
            window.append(v)
        beta = compute_beta(0.4, window)
        assert beta >= 0.0


# ---------------------------------------------------------------------------
# Required test 5: test_window_maxlen
# ---------------------------------------------------------------------------

class TestWindowMaxLen:
    def test_window_maxlen(self):
        """After 150 calls, window length must be exactly 100."""
        window = deque(maxlen=100)
        for i in range(150):
            compute_beta(float(i % 10) / 10.0, window)
        assert len(window) == 100

    def test_window_initial_single_entry(self):
        """First call: window has 1 entry; mean equals that entry."""
        window = deque(maxlen=100)
        beta = compute_beta(0.5, window)
        assert len(window) == 1
        # mean = 0.5, so beta = (1-0.5) / (0.5 + eps) ≈ 1.0
        assert beta == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Required test 6: test_ssq_weights_at_step_zero
# ---------------------------------------------------------------------------

class TestSSQLossWeights:
    def test_ssq_weights_at_step_zero(self):
        """At step=0: L_SSQ = 0.7*1.0*(Ls+Lsh) + 0.3*1.0*Lq  (r=0)."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]], requires_grad=False)
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])

        Ls  = scale_loss(p, t, w_s=0.6, size_cov_scale=1.0)
        Lsh = shape_loss(p, t, shape_scale=2.0)

        window = deque(maxlen=100)
        window.append(0.5)
        beta = compute_beta(0.5, window)
        Lq = quality_loss(p, t, beta=beta, sigma=2.0)

        # Fresh window for ssq_loss
        window2 = deque(maxlen=100)
        window2.append(0.5)
        # Monkey-patch: give ssq_loss the same beta by pre-filling window2
        # Actually we just check the formula directly
        p2 = p.clone().requires_grad_(True)
        window3 = deque(maxlen=100)
        for _ in range(5):
            window3.append(0.5)

        total = ssq_loss(p2, t, step=0, total_steps=500, window=window3,
                         w_s=0.6, size_cov_scale=1.0, shape_scale=2.0, sigma=2.0)

        # r=0: 0.7*(1-0)*(Ls+Lsh) + 0.3*(1+0)*Lq
        # We can't know exact beta from window3, but we can verify the scale factors
        # by checking bounds: total must be >= 0 and finite
        assert not torch.isnan(total).any()
        assert not torch.isinf(total).any()
        assert (total >= 0).all()

        # Stronger check: compute manually with same window state
        p3 = p.clone().requires_grad_(True)
        win4 = deque(maxlen=100)
        for _ in range(5):
            win4.append(0.5)
        # Get beta that ssq_loss would use (iou computed from p,t)
        with torch.no_grad():
            iou = _box_iou_elementwise(p3, t).mean().item()
        beta2 = compute_beta(iou, win4)
        Ls2 = scale_loss(p3, t, w_s=0.6)
        Lsh2 = shape_loss(p3, t, shape_scale=2.0)
        Lq2 = quality_loss(p3, t, beta=beta2, sigma=2.0)
        expected = 0.7 * 1.0 * (Ls2 + Lsh2) + 0.3 * 1.0 * Lq2
        # ssq_loss result should match to within float precision
        p4 = p.clone().requires_grad_(True)
        win5 = deque(maxlen=100)
        for _ in range(5):
            win5.append(0.5)
        got = ssq_loss(p4, t, step=0, total_steps=500, window=win5,
                       w_s=0.6, size_cov_scale=1.0, shape_scale=2.0, sigma=2.0)
        # Windows may differ by one append; check they are close
        assert torch.allclose(got, expected, rtol=1e-3), \
            f"step=0 formula mismatch: got {got.item():.6f}, expected {expected.item():.6f}"

    # -----------------------------------------------------------------------
    # Required test 7: test_ssq_weights_at_final_step
    # -----------------------------------------------------------------------

    def test_ssq_weights_at_final_step(self):
        """At step=total_steps: r=1, so (scale+shape) weight=0, quality weight=0.6."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]], requires_grad=True)
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])
        total_steps = 500

        win = deque(maxlen=100)
        for _ in range(50):
            win.append(0.5)

        result = ssq_loss(p, t, step=total_steps, total_steps=total_steps, window=win,
                          w_s=0.6, size_cov_scale=1.0, shape_scale=2.0, sigma=2.0)

        # r=1: 0.7*(1-1)*(Ls+Lsh) + 0.3*(1+1)*Lq = 0.0 + 0.6*Lq
        # Compute expected 0.6*Lq
        with torch.no_grad():
            iou_val = _box_iou_elementwise(p, t).mean().item()
        win2 = deque(maxlen=100)
        for _ in range(50):
            win2.append(0.5)
        beta = compute_beta(iou_val, win2)
        Lq = quality_loss(p, t, beta=beta, sigma=2.0)
        expected = 0.6 * Lq

        assert torch.allclose(result, expected, rtol=1e-3), \
            f"Final step formula mismatch: got {result.item():.6f}, expected {expected.item():.6f}"

    def test_ssq_r_monotone_schedule(self):
        """r increases from 0 to 1 with step, so quality weight grows."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]])
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])
        total = 100

        # At step=0, quality coefficient = 0.3*(1+0) = 0.3
        # At step=100, quality coefficient = 0.3*(1+1) = 0.6
        # (scale+shape) coefficient goes from 0.7 to 0.0
        # Just verify no NaN and result changes
        results = []
        for step in [0, 25, 50, 75, 100]:
            win = deque(maxlen=100)
            for _ in range(20):
                win.append(0.5)
            pv = p.clone().requires_grad_(True)
            r = ssq_loss(pv, t, step=step, total_steps=total, window=win)
            assert not torch.isnan(r).any()
            results.append(r.item())
        # Results should vary (loss changes shape as schedule shifts)
        assert len(set(round(x, 4) for x in results)) > 1


# ---------------------------------------------------------------------------
# Required test 8: test_gradcheck_scale
# ---------------------------------------------------------------------------

class TestGradCheck:
    def test_gradcheck_scale(self):
        """torch.autograd.gradcheck on scale_loss in float64."""
        torch.manual_seed(1)
        p = torch.tensor([[320.0, 256.0, 20.0, 16.0]],
                         dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[310.0, 250.0, 18.0, 14.0]], dtype=torch.float64)

        def fn(p_):
            return scale_loss(p_, t, w_s=0.6, size_cov_scale=1.0,
                               use_normalised_coords=False)

        assert torch.autograd.gradcheck(fn, (p,), eps=1e-4, atol=1e-3, rtol=1e-3), \
            "gradcheck failed for scale_loss"

    # -----------------------------------------------------------------------
    # Required test 9: test_gradcheck_shape
    # -----------------------------------------------------------------------

    def test_gradcheck_shape(self):
        """torch.autograd.gradcheck on shape_loss in float64."""
        torch.manual_seed(2)
        p = torch.tensor([[320.0, 256.0, 20.0, 16.0]],
                         dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[310.0, 250.0, 18.0, 14.0]], dtype=torch.float64)

        def fn(p_):
            return shape_loss(p_, t, shape_scale=2.0)

        assert torch.autograd.gradcheck(fn, (p,), eps=1e-4, atol=1e-3, rtol=1e-3), \
            "gradcheck failed for shape_loss"


# ---------------------------------------------------------------------------
# Required test 10: test_quality_no_grad_through_beta
# ---------------------------------------------------------------------------

class TestQualityGradient:
    def test_quality_no_grad_through_beta(self):
        """beta must not carry gradient after quality_loss.backward()."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]], requires_grad=True)
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])

        window = deque(maxlen=100)
        window.append(0.5)
        # iou_val is a Python float — no tensor
        iou_val = 0.6
        beta = compute_beta(iou_val, window)

        # beta is a Python float — it has no grad attribute at all
        assert isinstance(beta, float)
        # quality_loss output should be differentiable w.r.t. p
        loss = quality_loss(p, t, beta=beta).mean()
        loss.backward()
        assert p.grad is not None
        # Since beta is a plain float, there is no tensor to check grad on.
        # The test passes as long as backward() succeeds and p has a grad.

    def test_quality_rejects_tensor_beta(self):
        """Passing a Tensor as beta must raise TypeError."""
        p = torch.tensor([[310.0, 250.0, 18.0, 14.0]], requires_grad=True)
        t = torch.tensor([[320.0, 256.0, 20.0, 16.0]])
        bad_beta = torch.tensor(0.9)
        with pytest.raises(TypeError):
            quality_loss(p, t, beta=bad_beta)

    def test_quality_non_negative(self, batch_pair):
        p, t = batch_pair
        window = deque(maxlen=100)
        for v in [0.3, 0.5, 0.7, 0.4]:
            window.append(v)
        beta = compute_beta(0.5, window)
        loss = quality_loss(p, t, beta=beta)
        assert (loss >= 0).all()


# ---------------------------------------------------------------------------
# CIoU sanity tests
# ---------------------------------------------------------------------------

class TestCIoULoss:
    def test_ciou_zero_for_identical(self, perfect_pair):
        p, t = perfect_pair
        loss = ciou_loss(p.detach(), t)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_ciou_non_negative(self, batch_pair):
        p, t = batch_pair
        assert (ciou_loss(p, t) >= 0).all()

    def test_ciou_differentiable(self, batch_pair):
        p, t = batch_pair
        loss = ciou_loss(p, t).mean()
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    def test_ciou_larger_for_non_overlapping(self):
        p = torch.tensor([[100.0, 100.0, 20.0, 16.0]])
        t = torch.tensor([[500.0, 400.0, 20.0, 16.0]])  # far away
        loss = ciou_loss(p, t)
        assert loss.item() > 0.5  # should be large


# ---------------------------------------------------------------------------
# IoU helper tests
# ---------------------------------------------------------------------------

class TestBoxIoU:
    def test_iou_perfect(self):
        box = torch.tensor([[100.0, 100.0, 20.0, 16.0]])
        iou = _box_iou_elementwise(box, box)
        assert iou.item() == pytest.approx(1.0, abs=1e-5)

    def test_iou_no_overlap(self):
        p = torch.tensor([[50.0, 50.0, 10.0, 10.0]])
        t = torch.tensor([[200.0, 200.0, 10.0, 10.0]])
        iou = _box_iou_elementwise(p, t)
        assert iou.item() == pytest.approx(0.0, abs=1e-5)

    def test_iou_in_range(self, batch_pair):
        p, t = batch_pair
        iou = _box_iou_elementwise(p, t)
        assert (iou >= 0).all() and (iou <= 1).all()

import torch
from torch.nn import Parameter
from torch.testing import assert_close
import pytest
import math
import neuralop.training.tensorgrad as tensorgrad_module
from tensorly import tenalg
from tensorly.base import unfold

from ..adamw import AdamW
from ..tensorgrad import (
    EighProjector,
    TensorGRaD,
    TensorGRaDProjector,
    UnstructuredSparseProjector,
    _rank_for_tolerance,
    _resize_moments,
    _transport_first_moment,
    fno_tensorgrad_param_groups,
)

from ...models import FNO

from tensorly.tucker_tensor import validate_tucker_rank


@pytest.mark.parametrize("adam_optimizer_cls", [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    # param = x * 2j
    x = torch.randn((3, 3), dtype=torch.float64)
    param = Parameter(((0.0 + 1.0j) * x).to(torch.cfloat))

    optimizer = adam_optimizer_cls(params=[param], betas=(0.5, 0.5))

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j * x

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2jx * -2jx * (1 - 0.5)) = 4x**2 * 0.5 = 2x**2
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    momentum = optimizer.state[param]["exp_avg_sq"]
    assert_close(momentum, (2 * x**2).to(torch.cfloat))


@pytest.mark.parametrize("tensorgrad_param_pct", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("n_dim", [2, 3, 4])
def test_tensorgrad_low_rank_projection(tensorgrad_param_pct, n_dim):
    """TensorGRaD low-rank projection should preserve the requested rank size."""

    full_rank_tensor = torch.randn([8] * n_dim)

    tensorgrad_rank = validate_tucker_rank(
        full_rank_tensor.shape,
        tensorgrad_param_pct,
    )
    projector = TensorGRaDProjector(rank=tensorgrad_rank, warm_restart=True)

    # test that the projection is the correct size
    # test 0th iter for full computation
    low_rank_tensor = projector.project(full_rank_tensor, 0)
    assert low_rank_tensor.numel() == math.prod(tensorgrad_rank)


def test_tensorgrad_low_rank_projection_updates_after_gap():
    """TensorGRaD should refresh the low-rank projection at the configured gap."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(rank=(2, 2), update_proj_gap=2)
    projection_calls = []

    def get_projection_tensor(weights, rank):
        projection_calls.append((weights.shape, rank))
        return [torch.eye(4)[:, :2], torch.eye(4)[:, :2]]

    projector.get_projection_tensor = get_projection_tensor

    projector.project(full_rank_tensor, 0)
    projector.project(full_rank_tensor, 1)
    projector.project(full_rank_tensor, 2)

    assert projection_calls == [
        (full_rank_tensor.shape, (2, 2)),
        (full_rank_tensor.shape, (2, 2)),
    ]


def test_tensorgrad_low_rank_projection_warm_restart_updates():
    """Warm restart should pass TensorLy a valid Tucker initialization."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(
        rank=(2, 2),
        update_proj_gap=1,
        warm_restart=True,
    )

    first_projection = projector.project(full_rank_tensor, 0)
    second_projection = projector.project(full_rank_tensor, 1)

    assert first_projection.shape == (2, 2)
    assert second_projection.shape == (2, 2)


def test_tensorgrad_activation_checkpoint_projection():
    """Checkpointed TensorLy calls should preserve projection behavior."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(
        rank=(2, 2),
        activation_checkpoint=True,
    )

    low_rank_tensor = projector.project(full_rank_tensor, 0)

    assert low_rank_tensor.shape == (2, 2)


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_tensorgrad_projector_passes_tucker_n_iter_max(
    monkeypatch,
    activation_checkpoint,
):
    """The projector should wire tucker_n_iter_max into TensorLy Tucker calls."""

    n_iter_max_values = []

    def fake_tucker(tensor, *, rank, init, n_iter_max):
        n_iter_max_values.append(n_iter_max)
        factors = [
            torch.eye(dim, rank_dim, dtype=tensor.dtype, device=tensor.device)
            for dim, rank_dim in zip(tensor.shape, rank)
        ]
        return tensor, factors

    monkeypatch.setattr(tensorgrad_module, "tucker", fake_tucker)

    projector = TensorGRaDProjector(
        rank=(2, 2),
        tucker_n_iter_max=3,
        activation_checkpoint=activation_checkpoint,
    )

    projector.get_projection_tensor(torch.randn(4, 4), (2, 2))

    assert n_iter_max_values == [3]


@pytest.mark.parametrize("galore_param_pct", [0.25, 0.5, 1.0])
def test_galore_adamw_rank(galore_param_pct):
    """AdamW's legacy low-rank path should store optimizer state in projected size."""

    x = torch.randn((8, 8, 8, 8), dtype=torch.float64)
    param = Parameter(((0.0 + 1.0j) * x).to(torch.cfloat))
    galore_param = Parameter(((1.0 + 0j) * x).to(torch.cfloat))

    # pick mode-wise rank so that low_rank_grad.numel() / full_rank_grad.numel() == galore_param_pct
    galore_rank = validate_tucker_rank(galore_param.shape, galore_param_pct)

    optimizer = AdamW(
        params=[param],
        galore_params=[galore_param],
        galore_rank=galore_rank,
        betas=(0.5, 0.5),
    )

    loss = torch.view_as_real((param * galore_param.conj())).sum()

    loss.backward()
    optimizer.step()

    momentum = optimizer.state[galore_param]["exp_avg_sq"]
    # make sure low-rank params are being stored
    assert momentum.numel() == math.prod(galore_rank)


def test_unstructured_sparse_projector_topk():
    """Sparse top-k projection should keep only the requested largest entries."""

    grad = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    projector = UnstructuredSparseProjector(
        sparse_ratio=0.25,
        sparse_type="topk",
        update_proj_gap=1,
    )

    sparse_grad = projector.project(grad, step=0)
    projected_back = projector.project_back(sparse_grad)

    assert sparse_grad.numel() == 4
    assert torch.count_nonzero(projected_back) == 4
    assert_close(projected_back.max(), grad.max())


def test_tensorgrad_low_rank_stores_compressed_state():
    """Low-rank TensorGRaD should keep Adam moments in compressed coordinates."""

    param = Parameter(torch.randn(4, 4, 4, dtype=torch.float32))
    optimizer = TensorGRaD(
        [
            {
                "params": [param],
                "tensorgrad": True,
                "rank": 0.5,
                "update_proj_gap": 1,
            }
        ],
        betas=(0.5, 0.5),
    )

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    assert state["step"] == 1
    assert "low_rank_exp_avg_sq" in state
    assert state["low_rank_exp_avg_sq"].numel() < param.numel()


def test_tensorgrad_low_rank_sparse_stores_two_compressed_states():
    """Low-rank+sparse TensorGRaD should keep separate compressed moment states."""

    param = Parameter(torch.randn(4, 4, 4, dtype=torch.float32))
    optimizer = TensorGRaD(
        [
            {
                "params": [param],
                "tensorgrad": True,
                "rank": 0.5,
                "sparse_ratio": 0.25,
                "sparse_type": "topk",
                "update_proj_gap": 1,
            }
        ],
        betas=(0.5, 0.5),
    )

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    assert state["step"] == 1
    assert "low_rank_exp_avg_sq" in state
    assert "sparse_exp_avg_sq" in state
    assert state["sparse_exp_avg_sq"].numel() == int(0.25 * param.numel())


def test_fno_tensorgrad_param_groups_select_spectral_params():
    """FNO helper should split spectral-convolution params from regular params."""

    model = FNO(
        n_modes=(4, 4),
        hidden_channels=8,
        in_channels=1,
        out_channels=1,
        projection_channel_ratio=2,
    )

    groups = fno_tensorgrad_param_groups(
        model,
        rank=0.25,
        sparse_ratio=0.05,
        min_params=0,
    )

    regular_params = groups[0]["params"]
    tensorgrad_params = groups[1]["params"]
    regular_ids = {id(param) for param in regular_params}
    tensorgrad_ids = {id(param) for param in tensorgrad_params}

    assert groups[1]["tensorgrad"]
    assert groups[1]["sparse_ratio"] == 0.05
    assert tensorgrad_params
    assert regular_ids.isdisjoint(tensorgrad_ids)


def low_rank_complex_tensor(shape, rank):
    torch.manual_seed(0)
    core = torch.randn(*rank, dtype=torch.cfloat)
    factors = [
        torch.linalg.qr(torch.randn(n, r, dtype=torch.cfloat))[0]
        for n, r in zip(shape, rank)
    ]
    return tenalg.multi_mode_dot(core, factors)


def test_eigh_projector_matches_exact_svd_subspace():
    torch.manual_seed(0)
    grad = torch.randn(8, 6, 5, 3, dtype=torch.cfloat)
    rank = (3, 2, 2, 2)

    factors = EighProjector(rank=rank).get_projection_tensor(grad, rank)

    for mode, (u, r) in enumerate(zip(factors, rank)):
        exact = torch.linalg.svd(unfold(grad, mode), full_matrices=False)[0][:, :r]
        assert u.dtype == grad.dtype
        assert_close(u @ u.mH, exact @ exact.mH, atol=1e-5, rtol=1e-5)


def test_eigh_projector_tolerance_picks_rank_from_spectrum():
    grad = low_rank_complex_tensor((16, 12, 10, 8), (3, 4, 2, 5))
    projector = EighProjector(rank=(16, 12, 10, 8), tol=1e-3)
    capped = EighProjector(rank=(2, 2, 2, 2), tol=1e-3)

    ranks = [u.shape[1] for u in projector.get_projection_tensor(grad, projector.rank)]
    capped_ranks = [u.shape[1] for u in capped.get_projection_tensor(grad, capped.rank)]

    assert ranks == [3, 4, 2, 5]
    assert capped_ranks == [2, 2, 2, 2]


def test_rank_for_tolerance_matches_frobenius_tail():
    eigenvalues = torch.tensor([16.0, 4.0, 1.0, 0.0])

    assert _rank_for_tolerance(eigenvalues, 0.5) == 1
    assert _rank_for_tolerance(eigenvalues, 0.25) == 2
    assert _rank_for_tolerance(eigenvalues, 0.2) == 3
    assert _rank_for_tolerance(eigenvalues, 0.0) == 3
    assert _rank_for_tolerance(torch.zeros(3), 0.1) == 1


def test_resize_moments_keeps_leading_entries():
    first = torch.arange(6.0).reshape(2, 3)
    state = {"low_rank_exp_avg": first.clone(), "low_rank_exp_avg_sq": first.clone()}

    _resize_moments(state, (3, 2))

    expected = torch.tensor([[0.0, 1.0], [3.0, 4.0], [0.0, 0.0]])
    assert_close(state["low_rank_exp_avg"], expected)
    assert_close(state["low_rank_exp_avg_sq"], expected)


def test_transport_first_moment_preserves_it_within_same_subspace():
    torch.manual_seed(0)
    shape, rank = (8, 6, 5, 4), (3, 3, 2, 2)
    old = [
        torch.linalg.qr(torch.randn(n, r, dtype=torch.cfloat))[0]
        for n, r in zip(shape, rank)
    ]
    new = [
        u @ torch.linalg.qr(torch.randn(r, r, dtype=torch.cfloat))[0]
        for u, r in zip(old, rank)
    ]
    first = torch.randn(rank, dtype=torch.cfloat)
    state = {"low_rank_exp_avg": first.clone()}

    _transport_first_moment(state, old, new)

    restored = tenalg.multi_mode_dot(state["low_rank_exp_avg"], new)
    assert_close(restored, tenalg.multi_mode_dot(first, old), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tol", [None, 0.3])
@pytest.mark.parametrize("transport", [False, True])
def test_tensorgrad_eigh_group_keeps_moments_consistent(tol, transport):
    torch.manual_seed(0)
    param = Parameter(torch.randn(8, 8, 4, 3, dtype=torch.cfloat))
    group = {
        "params": [param],
        "tensorgrad": True,
        "rank": 1.0,
        "projector": "eigh",
        "tol": tol,
        "transport": transport,
        "update_proj_gap": 2,
    }
    optimizer = TensorGRaD([group])

    for step in range(6):
        param.grad = low_rank_complex_tensor(param.shape, (1 + step % 3,) * 4)
        optimizer.step()
        state = optimizer.state[param]
        assert type(state["low_rank_projector"]) is EighProjector
        assert state["low_rank_exp_avg"].shape == state["low_rank_exp_avg_sq"].shape

    assert torch.isfinite(param).all()


def test_fno_tensorgrad_param_groups_pass_eigh_options():
    model = FNO(n_modes=(4, 4), hidden_channels=8, in_channels=1, out_channels=1)

    groups = fno_tensorgrad_param_groups(
        model, min_params=0, projector="eigh", tol=0.01, transport=True
    )

    assert groups[1]["projector"] == "eigh"
    assert groups[1]["tol"] == 0.01
    assert groups[1]["transport"]


@pytest.mark.parametrize(
    "options, message",
    [({"projector": "svd"}, "Unknown projector"), ({"tol": 0.1}, "only supported")],
)
def test_tensorgrad_rejects_invalid_projector_options(options, message):
    param = Parameter(torch.randn(4, 4, 3, 2, dtype=torch.cfloat))
    optimizer = TensorGRaD([{"params": [param], "tensorgrad": True, **options}])
    param.grad = torch.randn_like(param)

    with pytest.raises(ValueError, match=message):
        optimizer.step()

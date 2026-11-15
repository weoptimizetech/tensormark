"""Run with an isolated venv's python -I, from outside the source checkout."""

import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np


class NoRealTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("the smoke test must not import real PyTorch")
        return None


def main():
    assert importlib.util.find_spec("torch") is None, "use a venv without PyTorch"
    sys.meta_path.insert(0, NoRealTorch())
    import tensormark as tm
    from tensormark import torch as shim

    package = Path(tm.__file__).resolve()
    native = Path(tm._native.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert native.parent == package.parent, native
    assert native.name == "tensormark" + importlib.machinery.EXTENSION_SUFFIXES[0], native
    assert isinstance(tm._native.__loader__, importlib.machinery.ExtensionFileLoader)
    version = importlib.metadata.version("tensormark")
    # Expected version lives in the source pyproject.toml, so a stale wheel
    # (built from an older tree) still fails this check.
    pyproject = Path(sys.argv[0]).resolve().parent.parent / "pyproject.toml"
    expected = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert tm.__version__ == tm._native.__version__ == version == expected, (
        version, expected)
    assert Path(shim.__file__).resolve().parent == package.parent
    assert "torch" not in sys.modules

    errors = {}

    def check(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        assert np.isfinite(actual).all(), (name, actual)
        assert np.isfinite(expected).all(), (name, expected)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6,
                                   err_msg=name)
        errors[name] = float(np.max(np.abs(actual - expected)))

    tm.tape_clear()
    a = np.array([[1, -2, 3], [0.5, 4, -1]], dtype=np.float32)
    b = np.array([[2, -1], [0.25, 3], [-2, 0.5]], dtype=np.float32)
    av, bv = tm.param(a), tm.param(b)
    product = tm.matmul(av, bv)
    reference = a @ b
    check("matmul_forward", product.numpy(), reference)
    loss = tm.sum_all(product)
    tm.tape_backward(loss)
    check("matmul_grad_a", av.grad, np.ones_like(reference) @ b.T)
    check("matmul_grad_b", bv.grad, a.T @ np.ones_like(reference))
    tm.tape_clear()

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    target = np.array([[1], [0]], dtype=np.float32)
    w = np.array([[0.25, -0.5]], dtype=np.float32)
    bias = np.array([0], dtype=np.float32)
    layer = shim.nn.Linear(2, 1)
    layer.weight.numpy()[:] = w
    layer.bias.numpy()[:] = bias
    optimizer = shim.optim.SGD(layer.parameters(), lr=0.01)
    optimizer.zero_grad()
    prediction = layer(shim.from_numpy(x))
    expected_prediction = x @ w.T + bias
    check("shim_forward", prediction.numpy(), expected_prediction)
    loss = shim.nn.MSELoss()(prediction, shim.from_numpy(target))
    before = float(np.mean((expected_prediction - target) ** 2))
    check("shim_loss", np.asarray(loss.item()), np.asarray(before))
    loss.backward()
    d = 2 * (expected_prediction - target) / target.size
    dw, db = d.T @ x, d.sum(axis=0)
    check("shim_grad_weight", layer.weight.grad, dw)
    check("shim_grad_bias", layer.bias.grad, db)
    optimizer.step()
    check("shim_sgd_weight", layer.weight.numpy(), w - 0.01 * dw)
    check("shim_sgd_bias", layer.bias.numpy(), bias - 0.01 * db)
    tm.tape_clear()
    after = shim.nn.MSELoss()(layer(shim.from_numpy(x)), shim.from_numpy(target)).item()
    assert np.isfinite(after) and after < before, (before, after)
    tm.tape_clear()
    assert "torch" not in sys.modules
    print(json.dumps({
        "version": version, "package": str(package), "native": str(native),
        "numpy_version": np.__version__, "max_abs_errors": errors,
        "matmul": reference.tolist(), "sgd_loss_before": before,
        "sgd_loss_after": after, "real_torch_imported": False,
    }, indent=2))


if __name__ == "__main__":
    main()

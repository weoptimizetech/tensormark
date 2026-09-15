"""macOS/CoreML shim integration: tiny generated fixtures, no LLM required.

Run with the CoreML venv: python -m unittest discover -s tools/tests -p test_ane_runtime.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "darwin", "CoreML requires macOS")
class AneRuntimeTests(unittest.TestCase):
    def test_source_and_compiled_packages(self):
        import coremltools as ct
        import numpy as np
        import torch

        class Segment(torch.nn.Module):
            def __init__(self, kv):
                super().__init__()
                self.kv = kv

            def forward(self, x):
                hidden = 2 * x + 1
                if not self.kv:
                    return hidden
                k = x.reshape(4, 2, 4).transpose(0, 1)
                return hidden, k, k + 2

        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="tm-ane-test-") as directory:
            temp = Path(directory)
            packages = []
            for kv in (False, True):
                graph = torch.jit.trace(Segment(kv).eval(), torch.zeros(4, 8).half())
                options = {"outputs": [ct.TensorType(name=n) for n in ("h", "k_0", "v_0")]} if kv else {}
                model = ct.convert(
                    graph, inputs=[ct.TensorType(name="x", shape=(4, 8), dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS16, **options,
                )
                package = temp / ("kv.mlpackage" if kv else "hidden.mlpackage")
                model.save(str(package))
                packages.append(package)
            binary = temp / "test_ane_shim"
            subprocess.run([
                "clang++", "-std=c++23", "-O2", "-fobjc-arc", "-I", str(root / "tensormark"),
                str(root / "tensormark/tests/test_ane_shim.cpp"),
                str(root / "tensormark/ane_shim.mm"), "-o", str(binary),
                "-framework", "CoreML", "-framework", "Foundation",
            ], check=True)
            # The source-package case exercises use of compileModelAtURL's
            # returned URL; the compiled case exercises direct model loading.
            subprocess.run([str(binary), *map(str, packages)], check=True)
            for package in packages:
                subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(temp)], check=True)
            subprocess.run([str(binary), *(str(p.with_suffix(".mlmodelc")) for p in packages)], check=True)


if __name__ == "__main__":
    unittest.main()

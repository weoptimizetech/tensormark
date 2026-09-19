"""macOS/CoreML shim integration: tiny generated fixtures, no LLM required.

The package is built here (torch -> coremltools) and compiled by
xcrun coremlcompiler, so this is the only gate that executes the shim's
prediction path. It needs a CoreML-capable torch AND coremltools; without them
it cannot run, and there is no fallback fixture.

Run with the CoreML venv: python -m unittest discover -s tools/tests -p test_ane_runtime.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "tensormark/ane_shim.mm"
GATE = ROOT / "tensormark/tests/test_ane_shim.cpp"


@unittest.skipUnless(sys.platform == "darwin", "CoreML requires macOS")
class AneRuntimeTests(unittest.TestCase):
    def test_source_and_compiled_packages(self):
        import coremltools as ct
        import numpy as np
        import torch

        class Segment(torch.nn.Module):
            """The shim's whole output contract: hidden plus post-RoPE K and V."""

            def forward(self, x):
                hidden = 2 * x + 1
                k = x.reshape(4, 2, 4).transpose(0, 1)
                return hidden, k, k + 2

        with tempfile.TemporaryDirectory(prefix="tm-ane-test-") as directory:
            temp = Path(directory)
            graph = torch.jit.trace(Segment().eval(), torch.zeros(4, 8).half())
            model = ct.convert(
                graph, inputs=[ct.TensorType(name="x", shape=(4, 8), dtype=np.float16)],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS16,
                outputs=[ct.TensorType(name=n) for n in ("h", "k_0", "v_0")],
            )
            package = temp / "kv.mlpackage"
            model.save(str(package))
            binary = temp / "test_ane_shim"
            subprocess.run([
                "clang++", "-std=c++23", "-O2", "-fobjc-arc", "-I", str(ROOT / "tensormark"),
                str(GATE), str(SHIM), "-o", str(binary),
                "-framework", "CoreML", "-framework", "Foundation", "-framework", "Accelerate",
            ], check=True)
            # The model-free half runs whatever else is passed, so run it alone too.
            model_free = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            self.assertIn("model-free", model_free.stdout)
            # A source package is refused with the command to run, not loaded: the
            # CoreML runtime executes compiled programs only.
            refused = subprocess.run([str(binary), str(package)], capture_output=True, text=True)
            self.assertEqual(refused.returncode, 1, refused.stderr)
            self.assertIn("coremlcompiler", refused.stderr)
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(temp)], check=True)
            compiled = subprocess.run([str(binary), str(package.with_suffix(".mlmodelc"))],
                                      capture_output=True, text=True, check=True)
            self.assertIn("package contracts PASS", compiled.stdout)


if __name__ == "__main__":
    unittest.main()

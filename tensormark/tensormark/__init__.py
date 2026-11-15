import importlib.machinery
import importlib.util
import os as _os

from ._loader import find_native as _find_native

# Never hide a broken packaged extension by loading a development binary.
_path = _find_native(_os.path.dirname(__file__))
_loader = importlib.machinery.ExtensionFileLoader("tensormark.tensormark", _path)
_spec = importlib.util.spec_from_file_location("tensormark.tensormark", _path,
                                               loader=_loader)
_native = importlib.util.module_from_spec(_spec)
_loader.exec_module(_native)

# Re-export the full native API (tensor ops, layers, optimizers, autograd
# entry points) so `import tensormark` exposes everything the extension
# defines, without leaking loader internals like glob/importlib.
globals().update({
    _k: _v for _k, _v in vars(_native).items()
    if not _k.startswith("__") and _k not in ("glob", "importlib")
})
__version__ = _native.__version__

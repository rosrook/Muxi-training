# Compatibility shim for ms-swift 4.1.3 + transformers >=5.14
# ms-swift MSHub.patch_hub() does trainer.create_repo / upload_folder,
# which were re-exports removed from transformers.trainer in recent releases.
#
# IMPORTANT: do NOT eagerly import transformers.trainer here.
# A site.pth that imports trainer on every Python startup can hang on Metax
# (torchao / hub / CUDA init). This module only installs a lazy import hook.
#
# For transformers==5.2.* (ms-swift 4.1.x official pin): patch is usually a no-op
# because create_repo is still present — safe to leave enabled.
from __future__ import annotations

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec


def _patch_module(mod) -> None:
    try:
        import huggingface_hub
    except Exception as e:  # noqa: BLE001
        print(f"[patch_transformers_for_ms_swift] hub import skip: {e}")
        return
    fixed = []
    if not hasattr(mod, "create_repo"):
        mod.create_repo = huggingface_hub.create_repo
        fixed.append("create_repo")
    if not hasattr(mod, "upload_folder"):
        mod.upload_folder = huggingface_hub.upload_folder
        fixed.append("upload_folder")
    if fixed:
        print(
            f"[patch_transformers_for_ms_swift] restored on transformers.trainer: {', '.join(fixed)}"
        )


class _TrainerPatchFinder(MetaPathFinder):
    """When transformers.trainer is first imported, patch missing hub re-exports."""

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if fullname != "transformers.trainer":
            return None
        # Avoid recursion: temporarily remove ourselves
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        try:
            spec = find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        origin = spec.loader

        class _Wrap(Loader):
            def create_module(self, spec):  # noqa: ANN001
                if hasattr(origin, "create_module"):
                    return origin.create_module(spec)
                return None

            def exec_module(self, module):  # noqa: ANN001
                origin.exec_module(module)
                _patch_module(module)

        spec.loader = _Wrap()
        return spec


def _install() -> None:
    if any(isinstance(x, _TrainerPatchFinder) for x in sys.meta_path):
        return
    sys.meta_path.insert(0, _TrainerPatchFinder())


_install()

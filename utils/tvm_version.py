import tvm
from packaging import version


def _check_tvm_version_is_latest():
    current_version = version.parse(tvm.__version__)
    target_version = version.parse("0.20.0")

    return current_version >= target_version


LatestTVM = _check_tvm_version_is_latest()

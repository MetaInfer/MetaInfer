from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PREBUILT = sorted((ROOT / "prebuilt").glob("*.o"))
SOURCES = ["csrc/bindings.cpp"]
if PREBUILT:
    SOURCES.append("csrc/w8a8_dispatch.cpp")
else:
    SOURCES.append("csrc/w8a8_gemm_hip.hip")


setup(
    name="metainfer_w8a8_backend",
    ext_modules=[
        CUDAExtension(
            name="metainfer_w8a8_backend",
            sources=SOURCES,
            extra_objects=[str(path) for path in PREBUILT],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--offload-arch=gfx928"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
)

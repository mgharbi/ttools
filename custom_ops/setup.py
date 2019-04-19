import os
import platform
import re
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension
import torch as th

def generate_pybind_wrapper(path, headers, has_cuda):
  s = "#include <torch/extension.h>\n"
  if has_cuda:
    s += "#define HL_PT_CUDA\n"
  s += "#include <HalidePytorchHelpers.h>\n"
  for h in headers:
    s += "#include \"{}\"\n".format(os.path.splitext(h)[0]+".pytorch.h")
  s += "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
  for h in headers:
    name = os.path.splitext(h)[0]
    s += "  m.def(\"{}\", &{}_th_, \"Pytorch wrapper of the Halide pipeline {}\");\n".format(
      name, name, name)
  s += "}\n"
  with open(path, 'w') as fid:
    fid.write(s)


abs_path = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.join(abs_path, "build")
src_dir = os.path.join(abs_path, "src")

# Check Gradient Halide is available
halide_dir = os.path.join(abs_path, "gradient-halide")
if not os.path.exists(halide_dir):
  raise ValueError("Halide directory {} is invalid".format(halide_dir))

include_dirs = [os.path.join(abs_path, "build"), os.path.join(halide_dir, "include")]
compile_args = ["-std=c++11", "-g"]
if platform.system() == "Darwin":
  compile_args += ["-stdlib=libc++"]  # on osx libstdc++ causes trouble

re_cc = re.compile(r".*\.pytorch\.h")
hl_srcs = [f for f in os.listdir(build_dir) if re_cc.match(f)]

ext_name = "custom_ops"
hl_sources = []
hl_libs = []
hl_headers = []
for f in hl_srcs:
  # Add all Halide generated torch wrapper
  hl_src = os.path.join(build_dir, f)
  hl_sources.append(hl_src)

  # Add all Halide-generated libraries
  hl_lib = hl_src.split(".")[0] + ".a"
  hl_libs.append(hl_lib)

  hl_header = hl_src.split(".")[0] + ".h"
  hl_headers.append(os.path.basename(hl_header))

  # TODO: add cuda hl lib

wrapper_path = os.path.join(build_dir, "pybind_wrapper.cpp")
generate_pybind_wrapper(wrapper_path, hl_headers, True)
# generate_pybind_wrapper(wrapper_path, hl_headers, th.cuda.is_available())

sources = [wrapper_path]
# sources = hl_sources + [wrapper_path]


# if th.cuda.is_available():
from torch.utils.cpp_extension import CUDAExtension
extension = CUDAExtension(ext_name, sources,
                          extra_objects=hl_libs,
                          extra_compile_args=compile_args)
# else:
#   from torch.utils.cpp_extension import CppExtension
#   extension = CppExtension(ext_name, sources,
#                            extra_objects=hl_libs,
#                            extra_compile_args=compile_args)

# TODO: import all extensions in root lib

setup(name=ext_name,
      verbose=True,
      url="",
      author_email="mgharbi@adobe.com",
      author="Michael Gharbi",
      version="0.0.0",
      # ext_package="custom_ops",
      # packages=find_packages(),
      ext_modules=[extension],
      include_dirs=include_dirs,
      cmdclass={"build_ext": BuildExtension}
      )

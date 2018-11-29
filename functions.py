import torch as th
import custom_ops as ops

class Scatter2Gather(th.autograd.Function):
  """"""
  @staticmethod
  def forward(ctx, data):
    output = data.new()
    output.resize_as_(data)
    ops.scatter2gather_forward(data, output)
    return output

  @staticmethod
  def backward(ctx, d_output):
    d_data = d_output.new()
    d_data.resize_as_(d_output)
    ops.scatter2gather_forward(d_output, d_data)
    return d_data

def main():
  print(dir(ops))

if __name__ == "__main__":
  main()


import torch
from torch.autograd import Variable

threshold = Variable(torch.Tensor([1]), requires_grad=True)
input = Variable(torch.Tensor([-10, 3, 10]), requires_grad=False)

# output = (torch.nn.Tanhshrink()(input/threshold))*threshold
output = (torch.nn.Hardshrink(1.0)(input/threshold))*threshold
# output = torch.min(input,threshold)
# output = torch.min(input/threshold, torch.Tensor([1.0]))*threshold

output.backward(Variable(torch.ones(3)))

print('output : {}, input grad: {}, threshold grad: {}'.format(output, input.grad, threshold.grad))

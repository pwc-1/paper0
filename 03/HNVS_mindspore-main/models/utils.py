"""
models - utils
"""

import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net, context
from mindspore.common.initializer import One


# With the Direct Feedback implementation, this should act position-wise
# Take C-dim vector per position and output a scalar
# Base Model class with save and load methods
class BaseModel(nn.Cell):

	def __init__(self):
		super(BaseModel, self).__init__()

class MemoryValueModel(nn.Cell):
	def __init__(self, input_dim, name="memory_value"):
		super(MemoryValueModel, self).__init__()
		self.name = name
		self.fc1 = nn.Dense(input_dim, 32, activation=nn.ReLU())
		self.fc2 = nn.Dense(32, 32, activation=nn.ReLU())
		self.fc3 = nn.Dense(32, 1, activation=nn.ReLU())
	def construct(self,x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x

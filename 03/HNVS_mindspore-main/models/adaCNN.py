import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
import numpy as np
from models.utils import BaseModel, MemoryValueModel
import mindspore as ms

def Relu(x):
    x = ops.relu(x)
    return x

class celoss(nn.Cell):
    def __init__(self):
        super(celoss, self).__init__()
    def construct(self, x, y):
        return ops.cross_entropy(x, y)
def celoss(x, y):
	return ops.cross_entropy(x, y)

class adaCNNNet(nn.Cell):
    def __init__(self, name, layers, output_dim, is_training, csn):
        super(adaCNNNet, self).__init__()
        self.name = name
        self.layers = layers
        self.output_dim = output_dim
        self.is_training = is_training
        self.csn = csn
        self.gradients = {}

        self.conv_layers = nn.CellList([nn.Conv2d(1, 32, 3)]+[nn.Conv2d(32, 32, 3) for _ in range(layers-1)])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dense = nn.Dense((28-layers) * (28-layers) * 32, output_dim)

    def construct(self, x, csn_value):
        running_output = x
        for i in range(self.layers):
            conv = self.conv_layers[i](running_output)
            if self.csn and csn_value[f"conv_{i}"] is not None:
                conv += csn_value[f"conv_{i}"]
            relu = Relu(x=conv)
            grad_fn = mindspore.grad(Relu, (0))
            self.gradients[f"conv_{i}"] = grad_fn(conv)
            maxpool = self.maxpool(relu)
            running_output = maxpool

        running_output = ops.reshape(running_output,(-1, (28-self.layers) * (28-self.layers) * 32))
        output = self.dense(running_output)
        if self.csn and csn_value["logits"] is not None:
            output += csn_value["logits"] 
        return output, self.gradients


class adaCNNModel(nn.Cell):
    def __init__(self, name, num_classes=5, is_training=None, num_test_classes=None):
        super(adaCNNModel, self).__init__()
        self.name = name
        self.num_test_classes = num_test_classes
        
        if self.num_test_classes is not None:
            self.logit_mask = np.zeros([1, num_classes])
            for i in np.arange(num_test_classes):
              self.logit_mask[0, i] = 1
        else:
            self.logit_mask = np.ones([1, num_classes])
            self.num_test_classes = num_classes

        self.is_training = is_training

        self.cnn_train = adaCNNNet(name="cnn", layers=4, output_dim=num_classes, is_training=self.is_training, csn=False)
        self.memory_key_model = adaCNNNet(name="cnn", layers=2, output_dim=32, is_training=self.is_training, csn=False)
        self.cnn_test = adaCNNNet(name="cnn", layers=4, output_dim=num_classes, is_training=self.is_training, csn=False)
        self.dense = nn.Dense(num_classes, num_classes)
        self.num_classes = num_classes
        self.MemoryValueModel = MemoryValueModel(input_dim=self.num_classes)
        self.grad = ops.GradOperation()
        self.celoss = celoss

    def construct(self, metatrain_input_tensors):
        train_inputs = metatrain_input_tensors["train_inputs"]
        train_labels = metatrain_input_tensors["train_labels"]
        test_inputs = metatrain_input_tensors["test_inputs"]
        test_labels = metatrain_input_tensors["test_labels"]
        batch_size = train_inputs.shape[0]
        train_inputs = ops.reshape(train_inputs, (-1, 1, 28, 28))
        test_inputs = ops.reshape(test_inputs, (-1, 1, 28, 28))
        train_labels = ops.reshape(train_labels, (-1, self.num_classes))
        test_labels = ops.reshape(test_labels, (-1, self.num_classes))
        
        inputs = ops.cat([train_inputs, test_inputs], axis=0)
        labels = ops.cat([train_labels, test_labels], axis=0)
        
        logits_train, cnn_train_gradients = self.cnn_train(train_inputs, 0) 
        
        train_loss = ops.cross_entropy(logits_train.view(batch_size, -1), train_labels.view(batch_size, -1),reduction='none')
        
        train_predictions = ops.softmax(logits_train, axis=1).argmax(axis=1)

        train_accuracy = (train_predictions == train_labels.argmax(axis=1)).sum()*1.0/len(train_labels)
        # CSN Memory Matrix and other operations should be added here
        memory_key_model_output, memory_key_gradients = self.memory_key_model(inputs, 0)
        keys = ops.split(
			memory_key_model_output,
			[train_inputs.shape[0], test_inputs.shape[0]],
			axis=0,
		)
        train_keys = keys[0].view(batch_size, -1, 32)
        test_keys = keys[1].view(batch_size, -1, 32)
        
		# - Values
        #grad_fn = mindspore.grad(ops.cross_entropy)
        # logits_train：（40，5）Fp32 train_labels：（40，5)Fp32
        gradient = self.grad(self.celoss)(logits_train, train_labels)
        from mindspore.ops import stop_gradient
        gradient = stop_gradient(gradient)

        # from mindspore.common.initializer import One
        # cnn_train_gradients = {
        #     "conv_1": Tensor(shape = (40, 32, 27, 27), dtype=ms.float32, init=One()),
        #     "conv_2": Tensor(shape = (40, 32, 26, 26), dtype=ms.float32, init=One()),
        #     "conv_3": Tensor(shape = (40, 32, 25, 25), dtype=ms.float32, init=One()),
        # }
        # gradient = Tensor(shape = (40, 5), dtype=ms.float32, init=One())
        csn_gradients = {
            "conv_1": cnn_train_gradients["conv_1"].view(-1, 27 * 27 * 32, 1) * ops.expand_dims(gradient, axis=1),
            "conv_2": cnn_train_gradients["conv_2"].view(-1, 26 * 26 * 32, 1) * ops.expand_dims(gradient, axis=1),
            "conv_3": cnn_train_gradients["conv_3"].view(-1, 25 * 25 * 32, 1) * ops.expand_dims(gradient, axis=1),
            "logits": ops.expand_dims(gradient, axis=2) * ops.expand_dims(gradient, axis=1),
		}
        
        self.train_values = train_values = {
			"conv_1": self.MemoryValueModel(csn_gradients["conv_1"]).view(batch_size, -1, 27 * 27 * 32),
			"conv_2": self.MemoryValueModel(csn_gradients["conv_2"]).view(batch_size, -1, 26 * 26 * 32),
			"conv_3": self.MemoryValueModel(csn_gradients["conv_3"]).view(batch_size, -1, 25 * 25 * 32),
			# "conv_4": MemoryValueModel(csn_gradients["conv_4"]).view(batch_size, -1, 24 * 24 * 32]),
			"logits": self.MemoryValueModel(csn_gradients["logits"]).view(batch_size, -1, self.num_classes),
		}


		# Calculating Value for Test Key
        matmul = ops.MatMul(transpose_b=True)
        dotp = ops.zeros((batch_size, self.num_classes, self.num_classes), mindspore.float32)
        for i in range(batch_size):
            dotp[i] = matmul(test_keys[i], train_keys[i])
        #breakpoint()
        #dotp = matmul(test_keys.view(-1,32), train_keys)
        self.attention_weights = attention_weights = ops.softmax(dotp)
        csn_values = []
        for value in train_values.values():
            temp = value
            for i in range(batch_size):
                temp[i] = ops.matmul(attention_weights[i],value[i])
            csn_values.append(temp)
        csn = dict(zip(train_values.keys(), csn_values))
        self.csn = {
			"conv_0": None,
			"conv_1": csn["conv_1"].view(-1, 32, 27, 27),
			"conv_2": csn["conv_2"].view(-1, 32, 26, 26),
			"conv_3": csn["conv_3"].view(-1, 32, 25, 25),
			# "conv_4": csn["conv_4"].view(-1, 24, 24, 32]),
			"logits": csn["logits"].view(-1, self.num_classes),
		}     
        
        # csn_debug = {
		# 	"conv_0": None,
		# 	"conv_1": Tensor(shape = (40, 32, 27, 27), dtype=ms.float32, init=One()),
		# 	"conv_2": Tensor(shape = (40, 32, 26, 26), dtype=ms.float32, init=One()),
		# 	"conv_3": Tensor(shape = (40, 32, 25, 25), dtype=ms.float32, init=One()),
		# 	# "conv_4": csn["conv_4"].view(-1, 24, 24, 32]),
		# 	"logits": Tensor(shape = (40, 5), dtype=ms.float32, init=One()),
        # }
          

        # Finally, pass CSN values to adaCNNNet
        logits_test, cnn_test_gradients = self.cnn_test(test_inputs, self.csn)#csn_debug)
		# self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.cnn_test.logits))
		# self.optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.test_loss)
       # breakpoint()
        #test_predictions = (ops.mul(logits_test, Tensor(self.logit_mask))).argmax(axis=1)
        test_predictions = ops.softmax(logits_test, axis=1).argmax(axis=1)
        test_accuracy = (test_predictions == test_labels.argmax(axis=1)).sum()*1.0/len(test_labels)
        
        return logits_test, train_accuracy

# model = adaCNNModel(name="adaCNN")
# train_inputs = torch.randn(32, 1, 28, 28)
# train_labels = torch.randint(0, 5, (32,))
# test_inputs = torch.randn(16, 1, 28, 28)
# test_labels = torch.randint(0, 5, (16,))

# loss, accuracy = model(train_inputs, train_labels, test_inputs, test_labels)
# print(f"Loss: {loss.item()}, Accuracy: {accuracy.item()}")
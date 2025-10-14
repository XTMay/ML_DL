# 从零开始：用NumPy构建AI模型完整教程

## 课程目标
通过本课程，您将学会使用纯NumPy从零实现各种AI模型，深入理解神经网络的数学原理和底层机制。

---

## 第一章：基础知识准备

### 1.1 NumPy核心概念

```python
import numpy as np

# 数组创建与基本操作
a = np.array([[1, 2, 3], [4, 5, 6]])
print("形状:", a.shape)  # (2, 3)
print("维度:", a.ndim)   # 2

# 广播机制（Broadcasting）
b = np.array([1, 2, 3])
result = a + b  # 自动扩展b的维度
print(result)
```

### 1.2 重要的数学运算

```python
# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # 或 A @ B

# 元素级运算
element_wise = A * B  # 对应元素相乘

# 转置
A_T = A.T

# 求和操作
sum_all = np.sum(A)
sum_axis0 = np.sum(A, axis=0)  # 按列求和
sum_axis1 = np.sum(A, axis=1, keepdims=True)  # 按行求和，保持维度
```

---

## 第二章：激活函数实现

### 2.1 常用激活函数

```python
class ActivationFunctions:
    """激活函数集合"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid导数"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """双曲正切函数"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Tanh导数"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU导数"""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Leaky ReLU导数"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        """Softmax函数（用于多分类）"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### 2.2 测试激活函数

```python
# 测试代码
x = np.linspace(-5, 5, 100)
sigmoid_output = ActivationFunctions.sigmoid(x)
relu_output = ActivationFunctions.relu(x)
tanh_output = ActivationFunctions.tanh(x)

print("输入范围:", x.min(), "到", x.max())
print("Sigmoid输出范围:", sigmoid_output.min(), "到", sigmoid_output.max())
```

---

## 第三章：损失函数实现

### 3.1 常见损失函数

```python
class LossFunctions:
    """损失函数集合"""
    
    @staticmethod
    def mse(y_true, y_pred):
        """均方误差（Mean Squared Error）"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """MSE导数"""
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """二元交叉熵"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        """二元交叉熵导数"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """多分类交叉熵"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        """多分类交叉熵导数"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) / y_true.shape[0]
```

---

## 第四章：神经网络层的实现

### 4.1 全连接层（Dense Layer）

```python
class DenseLayer:
    """全连接层实现"""
    
    def __init__(self, input_size, output_size, activation='relu'):
        """
        初始化全连接层
        
        参数:
            input_size: 输入特征数
            output_size: 输出特征数
            activation: 激活函数类型
        """
        # He初始化（适用于ReLU）
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        self.activation = activation
        self.input = None
        self.output = None
        self.z = None  # 激活前的值
        
    def forward(self, input_data):
        """前向传播"""
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.bias
        
        # 应用激活函数
        if self.activation == 'relu':
            self.output = ActivationFunctions.relu(self.z)
        elif self.activation == 'sigmoid':
            self.output = ActivationFunctions.sigmoid(self.z)
        elif self.activation == 'tanh':
            self.output = ActivationFunctions.tanh(self.z)
        elif self.activation == 'softmax':
            self.output = ActivationFunctions.softmax(self.z)
        elif self.activation == 'linear':
            self.output = self.z
        else:
            raise ValueError(f"未知的激活函数: {self.activation}")
            
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        # 计算激活函数的梯度
        if self.activation == 'relu':
            activation_grad = ActivationFunctions.relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            activation_grad = ActivationFunctions.sigmoid_derivative(self.z)
        elif self.activation == 'tanh':
            activation_grad = ActivationFunctions.tanh_derivative(self.z)
        elif self.activation == 'softmax' or self.activation == 'linear':
            activation_grad = 1
        else:
            raise ValueError(f"未知的激活函数: {self.activation}")
        
        # 链式法则
        delta = output_gradient * activation_grad
        
        # 计算梯度
        weights_gradient = np.dot(self.input.T, delta)
        bias_gradient = np.sum(delta, axis=0, keepdims=True)
        input_gradient = np.dot(delta, self.weights.T)
        
        # 更新参数
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient
```

### 4.2 Dropout层

```python
class DropoutLayer:
    """Dropout正则化层"""
    
    def __init__(self, dropout_rate=0.5):
        """
        初始化Dropout层
        
        参数:
            dropout_rate: 丢弃概率
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
        
    def forward(self, input_data):
        """前向传播"""
        if self.training:
            # 训练模式：随机丢弃
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                          size=input_data.shape)
            return input_data * self.mask / (1 - self.dropout_rate)
        else:
            # 测试模式：不丢弃
            return input_data
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        if self.training:
            return output_gradient * self.mask / (1 - self.dropout_rate)
        else:
            return output_gradient
```

### 4.3 批归一化层（Batch Normalization）

```python
class BatchNormalization:
    """批归一化层"""
    
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        """
        初始化批归一化层
        
        参数:
            input_size: 输入特征数
            momentum: 移动平均动量
            epsilon: 数值稳定性参数
        """
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True
        
        # 缓存中间结果用于反向传播
        self.x_centered = None
        self.std = None
        self.x_norm = None
        
    def forward(self, input_data):
        """前向传播"""
        if self.training:
            # 计算批次统计量
            batch_mean = np.mean(input_data, axis=0, keepdims=True)
            batch_var = np.var(input_data, axis=0, keepdims=True)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # 归一化
            self.x_centered = input_data - batch_mean
            self.std = np.sqrt(batch_var + self.epsilon)
            self.x_norm = self.x_centered / self.std
        else:
            # 测试模式使用移动平均
            self.x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # 缩放和偏移
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        m = output_gradient.shape[0]
        
        # 计算gamma和beta的梯度
        gamma_gradient = np.sum(output_gradient * self.x_norm, axis=0, keepdims=True)
        beta_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # 计算输入梯度
        dx_norm = output_gradient * self.gamma
        dx_centered = dx_norm / self.std
        dmean = -np.sum(dx_centered, axis=0, keepdims=True)
        dvar = np.sum(dx_norm * self.x_centered, axis=0, keepdims=True) * -0.5 * (self.std ** -3)
        
        input_gradient = dx_centered + (dmean + 2 * self.x_centered * dvar) / m
        
        # 更新参数
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient
        
        return input_gradient
```

---

## 第五章：完整神经网络实现

### 5.1 神经网络类

```python
class NeuralNetwork:
    """完整的神经网络实现"""
    
    def __init__(self):
        """初始化神经网络"""
        self.layers = []
        self.loss_function = None
        self.loss_derivative = None
        
    def add(self, layer):
        """添加层"""
        self.layers.append(layer)
        
    def set_loss(self, loss_function, loss_derivative):
        """设置损失函数"""
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
        
    def forward(self, X):
        """前向传播"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_grad, learning_rate):
        """反向传播"""
        gradient = loss_grad
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
            
    def train(self, X_train, y_train, epochs, learning_rate, batch_size=32, 
              X_val=None, y_val=None, verbose=True):
        """
        训练神经网络
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            X_val: 验证数据
            y_val: 验证标签
            verbose: 是否打印训练信息
        """
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # 设置所有层为训练模式
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 计算损失
                loss = self.loss_function(y_batch, output)
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                loss_grad = self.loss_derivative(y_batch, output)
                self.backward(loss_grad, learning_rate)
            
            # 计算平均损失
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            
            # 计算训练准确率
            train_pred = self.predict(X_train)
            train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
            history['train_acc'].append(train_acc)
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))
                history['val_acc'].append(val_acc)
                
                # 设置为评估模式
                for layer in self.layers:
                    if hasattr(layer, 'training'):
                        layer.training = False
                val_output = self.forward(X_val)
                val_loss = self.loss_function(y_val, val_output)
                history['val_loss'].append(val_loss)
                
                # 恢复训练模式
                for layer in self.layers:
                    if hasattr(layer, 'training'):
                        layer.training = True
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """预测"""
        # 设置为评估模式
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        
        # 恢复训练模式
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        return predictions
```

---

## 第六章：实战案例 - MNIST手写数字识别

### 6.1 数据准备

```python
def load_and_preprocess_data():
    """
    加载和预处理MNIST数据
    注：这里使用模拟数据，实际应用中需要下载真实MNIST数据
    """
    # 模拟数据生成（实际应用中替换为真实数据加载）
    np.random.seed(42)
    X_train = np.random.randn(1000, 784) * 0.1
    y_train = np.random.randint(0, 10, 1000)
    X_test = np.random.randn(200, 784) * 0.1
    y_test = np.random.randint(0, 10, 200)
    
    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One-hot编码
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot

# 加载数据
X_train, y_train, X_test, y_test = load_and_preprocess_data()
print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
```

### 6.2 构建和训练模型

```python
# 创建神经网络
model = NeuralNetwork()

# 添加层
model.add(DenseLayer(784, 128, activation='relu'))
model.add(BatchNormalization(128))
model.add(DropoutLayer(0.3))

model.add(DenseLayer(128, 64, activation='relu'))
model.add(BatchNormalization(64))
model.add(DropoutLayer(0.3))

model.add(DenseLayer(64, 10, activation='softmax'))

# 设置损失函数
model.set_loss(
    LossFunctions.categorical_crossentropy,
    LossFunctions.categorical_crossentropy_derivative
)

# 训练模型
history = model.train(
    X_train, y_train,
    epochs=100,
    learning_rate=0.01,
    batch_size=32,
    X_val=X_test,
    y_val=y_test,
    verbose=True
)

# 评估模型
predictions = model.predict(X_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"\n测试集准确率: {accuracy:.4f}")
```

---

## 第七章：卷积神经网络（CNN）

### 7.1 卷积层实现

```python
class ConvLayer:
    """二维卷积层实现"""
    
    def __init__(self, n_filters, filter_size, input_shape, stride=1, padding=0):
        """
        初始化卷积层
        
        参数:
            n_filters: 卷积核数量
            filter_size: 卷积核大小
            input_shape: 输入形状 (channels, height, width)
            stride: 步长
            padding: 填充
        """
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        input_channels, input_height, input_width = input_shape
        
        # 初始化权重和偏置
        self.filters = np.random.randn(n_filters, input_channels, 
                                      filter_size, filter_size) * 0.1
        self.bias = np.zeros((n_filters, 1))
        
        # 计算输出形状
        self.output_height = (input_height - filter_size + 2 * padding) // stride + 1
        self.output_width = (input_width - filter_size + 2 * padding) // stride + 1
        
    def forward(self, input_data):
        """前向传播"""
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        # 添加填充
        if self.padding > 0:
            input_padded = np.pad(input_data, 
                                 ((0, 0), (0, 0), 
                                  (self.padding, self.padding), 
                                  (self.padding, self.padding)), 
                                 mode='constant')
        else:
            input_padded = input_data
        
        # 初始化输出
        output = np.zeros((batch_size, self.n_filters, 
                          self.output_height, self.output_width))
        
        # 卷积操作
        for b in range(batch_size):
            for f in range(self.n_filters):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        receptive_field = input_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, f, i, j] = np.sum(receptive_field * self.filters[f]) + self.bias[f]
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        batch_size = self.input.shape[0]
        
        # 初始化梯度
        filters_gradient = np.zeros_like(self.filters)
        bias_gradient = np.zeros_like(self.bias)
        input_gradient = np.zeros_like(self.input)
        
        # 添加填充
        if self.padding > 0:
            input_padded = np.pad(self.input,
                                 ((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)),
                                 mode='constant')
            input_gradient_padded = np.pad(input_gradient,
                                          ((0, 0), (0, 0),
                                           (self.padding, self.padding),
                                           (self.padding, self.padding)),
                                          mode='constant')
        else:
            input_padded = self.input
            input_gradient_padded = input_gradient
        
        # 计算梯度
        for b in range(batch_size):
            for f in range(self.n_filters):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        receptive_field = input_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        filters_gradient[f] += output_gradient[b, f, i, j] * receptive_field
                        input_gradient_padded[b, :, h_start:h_end, w_start:w_end] += \
                            output_gradient[b, f, i, j] * self.filters[f]
                
                bias_gradient[f] += np.sum(output_gradient[:, f, :, :])
        
        # 更新参数
        self.filters -= learning_rate * filters_gradient / batch_size
        self.bias -= learning_rate * bias_gradient / batch_size
        
        # 移除填充
        if self.padding > 0:
            input_gradient = input_gradient_padded[:, :, 
                                                   self.padding:-self.padding,
                                                   self.padding:-self.padding]
        else:
            input_gradient = input_gradient_padded
        
        return input_gradient
```

### 7.2 池化层实现

```python
class MaxPoolingLayer:
    """最大池化层"""
    
    def __init__(self, pool_size=2, stride=2):
        """
        初始化池化层
        
        参数:
            pool_size: 池化窗口大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, input_data):
        """前向传播"""
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        pool_region = input_data[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(pool_region)
                        
                        # 保存最大值位置
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_indices[b, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        input_gradient = np.zeros_like(self.input)
        batch_size, channels, output_height, output_width = output_gradient.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        input_gradient[b, c, h_idx, w_idx] += output_gradient[b, c, i, j]
        
        return input_gradient
```

### 7.3 展平层

```python
class FlattenLayer:
    """展平层（将多维输入展平为二维）"""
    
    def forward(self, input_data):
        """前向传播"""
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]
        return input_data.reshape(batch_size, -1)
    
    def backward(self, output_gradient, learning_rate):
        """反向传播"""
        return output_gradient.reshape(self.input_shape)
```

---

## 第八章：循环神经网络（RNN）

### 8.1 基础RNN实现

```python
class RNNCell:
    """基础RNN单元"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        # Xavier初始化
        self.Wxh = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # 缓存用于反向传播
        self.cache = {}
        
    def forward(self, inputs, h_prev):
        """
        前向传播
        
        参数:
            inputs: 输入序列 (batch_size, seq_length, input_size)
            h_prev: 初始隐藏状态 (batch_size, hidden_size)
        
        返回:
            outputs: 输出序列 (batch_size, seq_length, output_size)
            h_states: 所有时间步的隐藏状态
        """
        batch_size, seq_length, _ = inputs.shape
        hidden_size = self.Whh.shape[0]
        output_size = self.Why.shape[1]
        
        # 初始化输出和隐藏状态
        outputs = np.zeros((batch_size, seq_length, output_size))
        h_states = np.zeros((batch_size, seq_length + 1, hidden_size))
        h_states[:, 0, :] = h_prev
        
        # 逐时间步前向传播
        for t in range(seq_length):
            x_t = inputs[:, t, :]
            h_prev = h_states[:, t, :]
            
            # 计算新的隐藏状态
            h_t = np.tanh(np.dot(x_t, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
            
            # 计算输出
            y_t = np.dot(h_t, self.Why) + self.by
            
            outputs[:, t, :] = y_t
            h_states[:, t + 1, :] = h_t
        
        # 保存用于反向传播
        self.cache = {
            'inputs': inputs,
            'h_states': h_states,
            'outputs': outputs
        }
        
        return outputs, h_states[:, -1, :]
    
    def backward(self, doutputs, learning_rate):
        """
        反向传播
        
        参数:
            doutputs: 输出梯度 (batch_size, seq_length, output_size)
            learning_rate: 学习率
        
        返回:
            dinputs: 输入梯度
            dh_prev: 初始隐藏状态梯度
        """
        inputs = self.cache['inputs']
        h_states = self.cache['h_states']
        
        batch_size, seq_length, input_size = inputs.shape
        hidden_size = self.Whh.shape[0]
        
        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dinputs = np.zeros_like(inputs)
        dh_next = np.zeros((batch_size, hidden_size))
        
        # 反向传播穿过时间（BPTT）
        for t in reversed(range(seq_length)):
            # 输出层梯度
            dy = doutputs[:, t, :]
            dWhy += np.dot(h_states[:, t + 1, :].T, dy)
            dby += np.sum(dy, axis=0, keepdims=True)
            
            # 隐藏层梯度
            dh = np.dot(dy, self.Why.T) + dh_next
            
            # tanh梯度
            dh_raw = (1 - h_states[:, t + 1, :] ** 2) * dh
            
            # 参数梯度
            dWxh += np.dot(inputs[:, t, :].T, dh_raw)
            dWhh += np.dot(h_states[:, t, :].T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            # 输入和隐藏状态梯度
            dinputs[:, t, :] = np.dot(dh_raw, self.Wxh.T)
            dh_next = np.dot(dh_raw, self.Whh.T)
        
        # 梯度裁剪（防止梯度爆炸）
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 更新参数
        self.Wxh -= learning_rate * dWxh / batch_size
        self.Whh -= learning_rate * dWhh / batch_size
        self.Why -= learning_rate * dWhy / batch_size
        self.bh -= learning_rate * dbh / batch_size
        self.by -= learning_rate * dby / batch_size
        
        return dinputs, dh_next
```

### 8.2 LSTM实现

```python
class LSTMCell:
    """长短期记忆网络（LSTM）单元"""
    
    def __init__(self, input_size, hidden_size):
        """
        初始化LSTM单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门、遗忘门、输出门、候选记忆的权重
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        
        self.cache = {}
        
    def forward(self, inputs, h_prev, c_prev):
        """
        前向传播
        
        参数:
            inputs: 输入序列 (batch_size, seq_length, input_size)
            h_prev: 初始隐藏状态 (batch_size, hidden_size)
            c_prev: 初始细胞状态 (batch_size, hidden_size)
        
        返回:
            outputs: 隐藏状态序列
            h_final: 最终隐藏状态
            c_final: 最终细胞状态
        """
        batch_size, seq_length, _ = inputs.shape
        
        # 初始化状态存储
        h_states = np.zeros((batch_size, seq_length + 1, self.hidden_size))
        c_states = np.zeros((batch_size, seq_length + 1, self.hidden_size))
        h_states[:, 0, :] = h_prev
        c_states[:, 0, :] = c_prev
        
        # 存储中间值
        f_gates = np.zeros((batch_size, seq_length, self.hidden_size))
        i_gates = np.zeros((batch_size, seq_length, self.hidden_size))
        o_gates = np.zeros((batch_size, seq_length, self.hidden_size))
        c_tildes = np.zeros((batch_size, seq_length, self.hidden_size))
        
        # 逐时间步前向传播
        for t in range(seq_length):
            x_t = inputs[:, t, :]
            h_prev = h_states[:, t, :]
            c_prev = c_states[:, t, :]
            
            # 拼接输入和隐藏状态
            concat = np.concatenate([x_t, h_prev], axis=1)
            
            # 遗忘门
            f_t = ActivationFunctions.sigmoid(np.dot(concat, self.Wf) + self.bf)
            
            # 输入门
            i_t = ActivationFunctions.sigmoid(np.dot(concat, self.Wi) + self.bi)
            
            # 候选记忆
            c_tilde_t = np.tanh(np.dot(concat, self.Wc) + self.bc)
            
            # 更新细胞状态
            c_t = f_t * c_prev + i_t * c_tilde_t
            
            # 输出门
            o_t = ActivationFunctions.sigmoid(np.dot(concat, self.Wo) + self.bo)
            
            # 更新隐藏状态
            h_t = o_t * np.tanh(c_t)
            
            # 保存状态
            h_states[:, t + 1, :] = h_t
            c_states[:, t + 1, :] = c_t
            f_gates[:, t, :] = f_t
            i_gates[:, t, :] = i_t
            o_gates[:, t, :] = o_t
            c_tildes[:, t, :] = c_tilde_t
        
        # 缓存用于反向传播
        self.cache = {
            'inputs': inputs,
            'h_states': h_states,
            'c_states': c_states,
            'f_gates': f_gates,
            'i_gates': i_gates,
            'o_gates': o_gates,
            'c_tildes': c_tildes
        }
        
        return h_states[:, 1:, :], h_states[:, -1, :], c_states[:, -1, :]
    
    def backward(self, dh, learning_rate):
        """
        反向传播
        
        参数:
            dh: 隐藏状态梯度 (batch_size, seq_length, hidden_size)
            learning_rate: 学习率
        """
        inputs = self.cache['inputs']
        h_states = self.cache['h_states']
        c_states = self.cache['c_states']
        f_gates = self.cache['f_gates']
        i_gates = self.cache['i_gates']
        o_gates = self.cache['o_gates']
        c_tildes = self.cache['c_tildes']
        
        batch_size, seq_length, _ = inputs.shape
        
        # 初始化梯度
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        
        dinputs = np.zeros_like(inputs)
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # BPTT
        for t in reversed(range(seq_length)):
            dh_t = dh[:, t, :] + dh_next
            
            # 输出门梯度
            do_t = dh_t * np.tanh(c_states[:, t + 1, :])
            do_raw = do_t * o_gates[:, t, :] * (1 - o_gates[:, t, :])
            
            # 细胞状态梯度
            dc_t = dh_t * o_gates[:, t, :] * (1 - np.tanh(c_states[:, t + 1, :]) ** 2) + dc_next
            
            # 候选记忆梯度
            dc_tilde_t = dc_t * i_gates[:, t, :]
            dc_tilde_raw = dc_tilde_t * (1 - c_tildes[:, t, :] ** 2)
            
            # 输入门梯度
            di_t = dc_t * c_tildes[:, t, :]
            di_raw = di_t * i_gates[:, t, :] * (1 - i_gates[:, t, :])
            
            # 遗忘门梯度
            df_t = dc_t * c_states[:, t, :]
            df_raw = df_t * f_gates[:, t, :] * (1 - f_gates[:, t, :])
            
            # 拼接输入
            concat = np.concatenate([inputs[:, t, :], h_states[:, t, :]], axis=1)
            
            # 累积权重梯度
            dWf += np.dot(concat.T, df_raw)
            dWi += np.dot(concat.T, di_raw)
            dWo += np.dot(concat.T, do_raw)
            dWc += np.dot(concat.T, dc_tilde_raw)
            
            dbf += np.sum(df_raw, axis=0, keepdims=True)
            dbi += np.sum(di_raw, axis=0, keepdims=True)
            dbo += np.sum(do_raw, axis=0, keepdims=True)
            dbc += np.sum(dc_tilde_raw, axis=0, keepdims=True)
            
            # 计算输入和隐藏状态梯度
            dconcat = (np.dot(df_raw, self.Wf.T) + 
                      np.dot(di_raw, self.Wi.T) + 
                      np.dot(do_raw, self.Wo.T) + 
                      np.dot(dc_tilde_raw, self.Wc.T))
            
            dinputs[:, t, :] = dconcat[:, :self.input_size]
            dh_next = dconcat[:, self.input_size:]
            dc_next = dc_t * f_gates[:, t, :]
        
        # 梯度裁剪
        for dparam in [dWf, dWi, dWo, dWc]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 更新参数
        self.Wf -= learning_rate * dWf / batch_size
        self.Wi -= learning_rate * dWi / batch_size
        self.Wo -= learning_rate * dWo / batch_size
        self.Wc -= learning_rate * dWc / batch_size
        self.bf -= learning_rate * dbf / batch_size
        self.bi -= learning_rate * dbi / batch_size
        self.bo -= learning_rate * dbo / batch_size
        self.bc -= learning_rate * dbc / batch_size
        
        return dinputs
```

---

## 第九章：优化器实现

### 9.1 SGD、Momentum、Adam优化器

```python
class Optimizer:
    """优化器基类"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def update(self, weights, bias, weights_grad, bias_grad):
        weights -= self.learning_rate * weights_grad
        bias -= self.learning_rate * bias_grad
        return weights, bias


class Momentum(Optimizer):
    """动量优化器"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = {}
        self.velocity_b = {}
        
    def update(self, layer_id, weights, bias, weights_grad, bias_grad):
        # 初始化速度
        if layer_id not in self.velocity_w:
            self.velocity_w[layer_id] = np.zeros_like(weights)
            self.velocity_b[layer_id] = np.zeros_like(bias)
        
        # 更新速度
        self.velocity_w[layer_id] = (self.momentum * self.velocity_w[layer_id] - 
                                     self.learning_rate * weights_grad)
        self.velocity_b[layer_id] = (self.momentum * self.velocity_b[layer_id] - 
                                     self.learning_rate * bias_grad)
        
        # 更新参数
        weights += self.velocity_w[layer_id]
        bias += self.velocity_b[layer_id]
        
        return weights, bias


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}  # 一阶矩估计
        self.v_w = {}  # 二阶矩估计
        self.m_b = {}
        self.v_b = {}
        self.t = 0     # 时间步
        
    def update(self, layer_id, weights, bias, weights_grad, bias_grad):
        # 初始化矩估计
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(weights)
            self.v_w[layer_id] = np.zeros_like(weights)
            self.m_b[layer_id] = np.zeros_like(bias)
            self.v_b[layer_id] = np.zeros_like(bias)
        
        self.t += 1
        
        # 更新权重的矩估计
        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * weights_grad
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (weights_grad ** 2)
        
        # 偏差修正
        m_w_hat = self.m_w[layer_id] / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w[layer_id] / (1 - self.beta2 ** self.t)
        
        # 更新偏置的矩估计
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * bias_grad
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (bias_grad ** 2)
        
        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)
        
        # 更新参数
        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return weights, bias
```

---

## 第十章：高级技术

### 10.1 学习率调度器

```python
class LearningRateScheduler:
    """学习率调度器"""
    
    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """阶梯衰减"""
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """指数衰减"""
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr, epoch, T_max=50):
        """余弦退火"""
        return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
    
    @staticmethod
    def warmup_cosine(initial_lr, epoch, warmup_epochs=5, T_max=50):
        """预热+余弦退火"""
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            return LearningRateScheduler.cosine_annealing(
                initial_lr, epoch - warmup_epochs, T_max - warmup_epochs
            )
```

### 10.2 数据增强

```python
class DataAugmentation:
    """数据增强工具"""
    
    @staticmethod
    def add_noise(X, noise_level=0.1):
        """添加高斯噪声"""
        noise = np.random.randn(*X.shape) * noise_level
        return X + noise
    
    @staticmethod
    def random_flip(X, axis=1):
        """随机翻转"""
        mask = np.random.rand(X.shape[0]) > 0.5
        X_aug = X.copy()
        X_aug[mask] = np.flip(X[mask], axis=axis)
        return X_aug
    
    @staticmethod
    def mixup(X, y, alpha=0.2):
        """Mixup数据增强"""
        batch_size = X.shape[0]
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = lam.reshape(-1, 1)
        
        indices = np.random.permutation(batch_size)
        X_mixed = lam * X + (1 - lam) * X[indices]
        
        if len(y.shape) > 1:
            lam = lam.reshape(-1, 1)
        y_mixed = lam * y + (1 - lam) * y[indices]
        
        return X_mixed, y_mixed
```

### 10.3 正则化技术

```python
class Regularization:
    """正则化工具"""
    
    @staticmethod
    def l1_regularization(weights, lambda_reg):
        """L1正则化（Lasso）"""
        return lambda_reg * np.sum(np.abs(weights))
    
    @staticmethod
    def l1_gradient(weights, lambda_reg):
        """L1正则化梯度"""
        return lambda_reg * np.sign(weights)
    
    @staticmethod
    def l2_regularization(weights, lambda_reg):
        """L2正则化（Ridge）"""
        return 0.5 * lambda_reg * np.sum(weights ** 2)
    
    @staticmethod
    def l2_gradient(weights, lambda_reg):
        """L2正则化梯度"""
        return lambda_reg * weights
    
    @staticmethod
    def elastic_net(weights, lambda_l1, lambda_l2):
        """弹性网络（L1+L2）"""
        l1_term = lambda_l1 * np.sum(np.abs(weights))
        l2_term = 0.5 * lambda_l2 * np.sum(weights ** 2)
        return l1_term + l2_term
```

---

## 第十一章：实战案例2 - 情感分析RNN

### 11.1 文本预处理

```python
class TextPreprocessor:
    """文本预处理工具"""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def build_vocab(self, texts):
        """构建词汇表"""
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 添加到词汇表
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            self.word_to_idx[word] = self.vocab_size
            self.idx_to_word[self.vocab_size] = word
            self.vocab_size += 1
    
    def texts_to_sequences(self, texts):
        """将文本转换为序列"""
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_to_idx.get(word, 1) for word in words]
            sequences.append(sequence)
        return sequences
    
    def pad_sequences(self, sequences):
        """填充序列到固定长度"""
        padded = np.zeros((len(sequences), self.max_sequence_length), dtype=int)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_sequence_length)
            padded[i, :length] = seq[:length]
        return padded


# 示例使用
texts_train = [
    "this movie is great",
    "i love this film",
    "terrible movie waste of time",
    "amazing story and acting"
]
labels_train = np.array([1, 1, 0, 1])  # 1=正面, 0=负面

preprocessor = TextPreprocessor(max_vocab_size=1000, max_sequence_length=20)
preprocessor.build_vocab(texts_train)
X_train_seq = preprocessor.texts_to_sequences(texts_train)
X_train_padded = preprocessor.pad_sequences(X_train_seq)

print("词汇表大小:", preprocessor.vocab_size)
print("填充后序列形状:", X_train_padded.shape)
```

### 11.2 词嵌入层

```python
class EmbeddingLayer:
    """词嵌入层"""
    
    def __init__(self, vocab_size, embedding_dim):
        """
        初始化嵌入层
        
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 随机初始化嵌入矩阵
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
    def forward(self, input_indices):
        """
        前向传播
        
        参数:
            input_indices: 输入索引 (batch_size, sequence_length)
        
        返回:
            embedded: 嵌入向量 (batch_size, sequence_length, embedding_dim)
        """
        self.input_indices = input_indices
        return self.embeddings[input_indices]
    
    def backward(self, dembedded, learning_rate):
        """
        反向传播
        
        参数:
            dembedded: 嵌入层输出梯度 (batch_size, sequence_length, embedding_dim)
            learning_rate: 学习率
        """
        # 初始化嵌入梯度
        dembeddings = np.zeros_like(self.embeddings)
        
        # 累积梯度
        batch_size, seq_length, embed_dim = dembedded.shape
        for b in range(batch_size):
            for t in range(seq_length):
                idx = self.input_indices[b, t]
                dembeddings[idx] += dembedded[b, t]
        
        # 更新嵌入矩阵
        self.embeddings -= learning_rate * dembeddings
        
        return None  # 不需要返回输入梯度
```

---

## 第十二章：模型评估与可视化

### 12.1 评估指标

```python
class Metrics:
    """模型评估指标"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """准确率"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, average='binary'):
        """精确率"""
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp + 1e-10)
        else:
            # 多分类平均
            precisions = []
            for cls in np.unique(y_true):
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precisions.append(tp / (tp + fp + 1e-10))
            return np.mean(precisions)
    
    @staticmethod
    def recall(y_true, y_pred, average='binary'):
        """召回率"""
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn + 1e-10)
        else:
            recalls = []
            for cls in np.unique(y_true):
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recalls.append(tp / (tp + fn + 1e-10))
            return np.mean(recalls)
    
    @staticmethod
    def f1_score(y_true, y_pred, average='binary'):
        """F1分数"""
        precision = Metrics.precision(y_true, y_pred, average)
        recall = Metrics.recall(y_true, y_pred, average)
        return 2 * (precision * recall) / (precision + recall + 1e-10)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes):
        """混淆矩阵"""
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true)):
            matrix[y_true[i], y_pred[i]] += 1
        return matrix
    
    @staticmethod
    def roc_auc_score(y_true, y_scores):
        """ROC-AUC分数（二分类）"""
        # 简化实现
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            tpr = tp / (tp + fn + 1e-10)
            fpr = fp / (fp + tn + 1e-10)
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # 计算AUC（梯形法则）
        auc = 0
        for i in range(len(fpr_list) - 1):
            auc += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i] + tpr_list[i+1]) / 2
        
        return abs(auc)
```

### 12.2 模型保存与加载

```python
class ModelSerializer:
    """模型序列化工具"""
    
    @staticmethod
    def save_model(model, filepath):
        """保存模型参数"""
        model_params = {
            'layers': []
        }
        
        for i, layer in enumerate(model.layers):
            layer_params = {
                'type': layer.__class__.__name__,
                'params': {}
            }
            
            if hasattr(layer, 'weights'):
                layer_params['params']['weights'] = layer.weights
            if hasattr(layer, 'bias'):
                layer_params['params']['bias'] = layer.bias
            if hasattr(layer, 'gamma'):
                layer_params['params']['gamma'] = layer.gamma
            if hasattr(layer, 'beta'):
                layer_params['params']['beta'] = layer.beta
            if hasattr(layer, 'running_mean'):
                layer_params['params']['running_mean'] = layer.running_mean
            if hasattr(layer, 'running_var'):
                layer_params['params']['running_var'] = layer.running_var
            
            model_params['layers'].append(layer_params)
        
        np.save(filepath, model_params, allow_pickle=True)
        print(f"模型已保存到: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """加载模型参数"""
        model_params = np.load(filepath, allow_pickle=True).item()
        print(f"模型已从 {filepath} 加载")
        return model_params
```

### 12.3 训练过程可视化

```python
class TrainingVisualizer:
    """训练过程可视化"""
    
    @staticmethod
    def plot_history(history):
        """
        绘制训练历史（需要matplotlib）
        
        参数:
            history: 训练历史字典
        """
        print("\n训练历史统计:")
        print("=" * 50)
        
        if 'train_loss' in history:
            print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
        if 'train_acc' in history:
            print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
        if 'val_acc' in history:
            print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
        
        print("=" * 50)
    
    @staticmethod
    def print_confusion_matrix(cm, class_names=None):
        """打印混淆矩阵"""
        print("\n混淆矩阵:")
        print("=" * 50)
        
        if class_names is None:
            class_names = [f"类别{i}" for i in range(cm.shape[0])]
        
        # 打印表头
        print(f"{'真实\\预测':<15}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        print("-" * 50)
        
        # 打印每一行
        for i, name in enumerate(class_names):
            print(f"{name:<15}", end="")
            for j in range(cm.shape[1]):
                print(f"{cm[i,j]:>10}", end="")
            print()
        print("=" * 50)
```

---

## 第十三章：注意力机制（Attention Mechanism）

### 13.1 基础注意力实现

```python
class AttentionLayer:
    """注意力机制层"""
    
    def __init__(self, hidden_size):
        """
        初始化注意力层
        
        参数:
            hidden_size: 隐藏层维度
        """
        self.hidden_size = hidden_size
        
        # 注意力权重矩阵
        self.W_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v_a = np.random.randn(hidden_size, 1) * 0.01
        
    def forward(self, hidden_states, query):
        """
        前向传播
        
        参数:
            hidden_states: 编码器隐藏状态 (batch_size, seq_length, hidden_size)
            query: 查询向量（解码器状态） (batch_size, hidden_size)
        
        返回:
            context: 上下文向量 (batch_size, hidden_size)
            attention_weights: 注意力权重 (batch_size, seq_length)
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # 扩展query维度
        query_expanded = np.expand_dims(query, 1)  # (batch_size, 1, hidden_size)
        query_repeated = np.repeat(query_expanded, seq_length, axis=1)
        
        # 计算注意力分数
        # score = tanh(W_a * hidden_states + U_a * query)
        scores = np.tanh(
            np.dot(hidden_states, self.W_a) + np.dot(query_repeated, self.U_a)
        )  # (batch_size, seq_length, hidden_size)
        
        # 计算注意力权重
        scores = np.dot(scores, self.v_a).squeeze(-1)  # (batch_size, seq_length)
        attention_weights = ActivationFunctions.softmax(scores.reshape(batch_size, -1))
        
        # 计算上下文向量
        attention_weights_expanded = np.expand_dims(attention_weights, -1)
        context = np.sum(hidden_states * attention_weights_expanded, axis=1)
        
        # 缓存用于反向传播
        self.cache = {
            'hidden_states': hidden_states,
            'query': query,
            'attention_weights': attention_weights,
            'context': context
        }
        
        return context, attention_weights
    
    def backward(self, dcontext, learning_rate):
        """反向传播（简化版本）"""
        hidden_states = self.cache['hidden_states']
        attention_weights = self.cache['attention_weights']
        
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # 计算隐藏状态梯度
        attention_weights_expanded = np.expand_dims(attention_weights, -1)
        dhidden_states = dcontext[:, np.newaxis, :] * attention_weights_expanded
        
        return dhidden_states
```

### 13.2 多头注意力（Multi-Head Attention）

```python
class MultiHeadAttention:
    """多头注意力机制"""
    
    def __init__(self, hidden_size, num_heads):
        """
        初始化多头注意力
        
        参数:
            hidden_size: 隐藏层维度（必须能被num_heads整除）
            num_heads: 注意力头数量
        """
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V的投影矩阵
        self.W_q = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_k = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_v = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_o = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        
    def split_heads(self, x):
        """将输入分割成多个头"""
        batch_size, seq_length, hidden_size = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
    
    def merge_heads(self, x):
        """合并多个头"""
        batch_size, num_heads, seq_length, head_dim = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.hidden_size)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        
        参数:
            Q: 查询 (batch_size, num_heads, seq_length, head_dim)
            K: 键 (batch_size, num_heads, seq_length, head_dim)
            V: 值 (batch_size, num_heads, seq_length, head_dim)
            mask: 掩码
        """
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_length, seq_length)
        scores = scores / np.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Softmax归一化
        attention_weights = ActivationFunctions.softmax(scores)
        
        # 计算输出
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: 查询张量 (batch_size, seq_length, hidden_size)
            key: 键张量 (batch_size, seq_length, hidden_size)
            value: 值张量 (batch_size, seq_length, hidden_size)
            mask: 注意力掩码
        """
        batch_size = query.shape[0]
        
        # 线性投影
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 应用缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多个头
        attention_output = self.merge_heads(attention_output)
        
        # 最终线性投影
        output = np.dot(attention_output, self.W_o)
        
        # 缓存用于反向传播
        self.cache = {
            'query': query,
            'key': key,
            'value': value,
            'Q': Q,
            'K': K,
            'V': V,
            'attention_weights': attention_weights,
            'attention_output': attention_output
        }
        
        return output, attention_weights
```

---

## 第十四章：生成对抗网络（GAN）基础

### 14.1 简单GAN实现

```python
class Generator:
    """生成器网络"""
    
    def __init__(self, latent_dim, output_dim):
        """
        初始化生成器
        
        参数:
            latent_dim: 潜在空间维度
            output_dim: 输出维度
        """
        self.layers = []
        self.layers.append(DenseLayer(latent_dim, 128, activation='relu'))
        self.layers.append(DenseLayer(128, 256, activation='relu'))
        self.layers.append(DenseLayer(256, output_dim, activation='tanh'))
        
    def forward(self, z):
        """前向传播"""
        output = z
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient, learning_rate):
        """反向传播"""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


class Discriminator:
    """判别器网络"""
    
    def __init__(self, input_dim):
        """
        初始化判别器
        
        参数:
            input_dim: 输入维度
        """
        self.layers = []
        self.layers.append(DenseLayer(input_dim, 256, activation='relu'))
        self.layers.append(DropoutLayer(0.3))
        self.layers.append(DenseLayer(256, 128, activation='relu'))
        self.layers.append(DropoutLayer(0.3))
        self.layers.append(DenseLayer(128, 1, activation='sigmoid'))
        
    def forward(self, x):
        """前向传播"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient, learning_rate):
        """反向传播"""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


class GAN:
    """生成对抗网络"""
    
    def __init__(self, latent_dim, data_dim):
        """
        初始化GAN
        
        参数:
            latent_dim: 潜在空间维度
            data_dim: 数据维度
        """
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        
    def train_step(self, real_data, batch_size, learning_rate):
        """
        训练一步
        
        参数:
            real_data: 真实数据
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 训练判别器
        # 1. 真实数据
        real_labels = np.ones((batch_size, 1))
        d_real = self.discriminator.forward(real_data)
        d_loss_real = LossFunctions.binary_crossentropy(real_labels, d_real)
        d_grad_real = LossFunctions.binary_crossentropy_derivative(real_labels, d_real)
        self.discriminator.backward(d_grad_real, learning_rate)
        
        # 2. 生成数据
        noise = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(noise)
        fake_labels = np.zeros((batch_size, 1))
        
        d_fake = self.discriminator.forward(fake_data)
        d_loss_fake = LossFunctions.binary_crossentropy(fake_labels, d_fake)
        d_grad_fake = LossFunctions.binary_crossentropy_derivative(fake_labels, d_fake)
        self.discriminator.backward(d_grad_fake, learning_rate)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # 训练生成器
        noise = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(noise)
        
        # 生成器希望判别器认为生成的数据是真实的
        misleading_labels = np.ones((batch_size, 1))
        d_output = self.discriminator.forward(fake_data)
        g_loss = LossFunctions.binary_crossentropy(misleading_labels, d_output)
        g_grad = LossFunctions.binary_crossentropy_derivative(misleading_labels, d_output)
        
        # 反向传播通过判别器（不更新判别器参数）
        # 然后更新生成器
        self.generator.backward(g_grad, learning_rate)
        
        return d_loss, g_loss
```

---

## 第十五章：实战项目 - 完整图像分类CNN

### 15.1 完整CNN模型

```python
class CNNImageClassifier:
    """完整的CNN图像分类器"""
    
    def __init__(self, input_shape, num_classes):
        """
        初始化CNN分类器
        
        参数:
            input_shape: 输入形状 (channels, height, width)
            num_classes: 类别数量
        """
        self.layers = []
        
        # 第一个卷积块
        self.layers.append(ConvLayer(32, 3, input_shape, stride=1, padding=1))
        self.layers.append(MaxPoolingLayer(pool_size=2, stride=2))
        
        # 第二个卷积块
        conv2_input_shape = (32, input_shape[1]//2, input_shape[2]//2)
        self.layers.append(ConvLayer(64, 3, conv2_input_shape, stride=1, padding=1))
        self.layers.append(MaxPoolingLayer(pool_size=2, stride=2))
        
        # 展平层
        self.layers.append(FlattenLayer())
        
        # 全连接层
        flatten_size = 64 * (input_shape[1]//4) * (input_shape[2]//4)
        self.layers.append(DenseLayer(flatten_size, 128, activation='relu'))
        self.layers.append(DropoutLayer(0.5))
        self.layers.append(DenseLayer(128, num_classes, activation='softmax'))
        
    def forward(self, X):
        """前向传播"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_grad, learning_rate):
        """反向传播"""
        gradient = loss_grad
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train(self, X_train, y_train, epochs, learning_rate, batch_size=32):
        """训练模型"""
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 计算损失
                loss = LossFunctions.categorical_crossentropy(y_batch, output)
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                loss_grad = LossFunctions.categorical_crossentropy_derivative(y_batch, output)
                self.backward(loss_grad, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            
            # 计算准确率
            predictions = self.predict(X_train)
            accuracy = np.mean(predictions == np.argmax(y_train, axis=1))
            history['train_acc'].append(accuracy)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        return history
    
    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
```

---

## 第十六章：高级实践与技巧

### 16.1 梯度检查

```python
class GradientChecker:
    """梯度检查工具"""
    
    @staticmethod
    def numerical_gradient(f, x, epsilon=1e-5):
        """
        数值梯度计算
        
        参数:
            f: 函数
            x: 输入
            epsilon: 微小扰动
        """
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            old_value = x[idx]
            
            x[idx] = old_value + epsilon
            f_plus = f(x)
            
            x[idx] = old_value - epsilon
            f_minus = f(x)
            
            grad[idx] = (f_plus - f_minus) / (2 * epsilon)
            
            x[idx] = old_value
            it.iternext()
        
        return grad
    
    @staticmethod
    def check_gradient(analytical_grad, numerical_grad, threshold=1e-7):
        """
        检查解析梯度和数值梯度是否接近
        
        参数:
            analytical_grad: 解析梯度
            numerical_grad: 数值梯度
            threshold: 阈值
        """
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
        difference = numerator / (denominator + 1e-10)
        
        if difference < threshold:
            print(f"✓ 梯度检查通过! 差异: {difference:.2e}")
            return True
        else:
            print(f"✗ 梯度检查失败! 差异: {difference:.2e}")
            return False
```

### 16.2 早停与模型检查点

```python
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001):
        """
        初始化早停
        
        参数:
            patience: 容忍轮数
            min_delta: 最小改进量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """检查是否应该早停"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"早停触发！在{self.patience}轮后验证损失未改善")
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class ModelCheckpoint:
    """模型检查点"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        """
        初始化检查点
        
        参数:
            filepath: 保存路径
            monitor: 监控指标
            mode: 'min'或'max'
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        
    def __call__(self, model, current_value):
        """检查是否应该保存模型"""
        if self.mode == 'min':
            is_better = current_value < self.best_value
        else:
            is_better = current_value > self.best_value
        
        if is_better:
            self.best_value = current_value
            ModelSerializer.save_model(model, self.filepath)
            print(f"模型已保存: {self.monitor} = {current_value:.4f}")
```

---

## 第十七章：总结与进阶路线

### 17.1 课程总结

本课程涵盖了以下内容：

1. **基础组件**
   - 激活函数（Sigmoid, ReLU, Tanh, Softmax等）
   - 损失函数（MSE, 交叉熵等）
   - 优化器（SGD, Momentum, Adam）

2. **神经网络层**
   - 全连接层（Dense Layer）
   - 卷积层（Convolutional Layer）
   - 池化层（Pooling Layer）
   - Dropout层
   - 批归一化（Batch Normalization）

3. **模型架构**
   - 前馈神经网络（Feedforward Neural Network）
   - 卷积神经网络（CNN）
   - 循环神经网络（RNN/LSTM）
   - 注意力机制（Attention）
   - 生成对抗网络（GAN）

4. **训练技巧**
   - 数据增强
   - 正则化
   - 学习率调度
   - 早停
   - 梯度检查

### 17.2 进阶学习路线

**掌握本课程后，建议按以下路线继续学习：**

1. **Transformer架构**
   - 自注意力机制
   - 位置编码
   - 编码器-解码器结构

2. **高级优化技术**
   - AdaGrad, RMSProp
   - 学习率预热
   - 梯度累积

3. **模型压缩与加速**
   - 模型剪枝
   - 知识蒸馏
   - 量化技术

4. **实际应用**
   - 目标检测（YOLO, R-CNN）
   - 语义分割
   - 生成模型（VAE, Diffusion Models）
   - 强化学习基础

### 17.3 实践
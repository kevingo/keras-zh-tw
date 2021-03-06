# Keras 問與答: Keras 的常見問題

- [我要如何引用 Keras？](#我要如何引用-Keras？)
- [我要如何在 GPU 上執行 Keras？](#我要如何在-GPU-上執行-Keras？)
- ["樣本(sample)"、"批次(batch)" 和 "epoch(訓練週期)" 代表什麼意思？](#"樣本(sample)"、"批次(batch)"-和-"epoch(訓練週期)"-代表什麼意思？)
- [我該如何保存 Keras 訓練出來的模型？](#我該如何保存-Keras-訓練出來的模型？)
- [為什麼我在訓練階段的誤差(loss)比測試階段來得高？](#為什麼我在訓練階段的誤差(loss)比測試階段來得高？)
- [我要如何得到在訓練時中間層的輸出？](#我要如何得到在訓練時中間層的輸出？)
- [當資料量太大無法一次讀進去記憶體時該怎麼處理？](#當資料量太大無法一次讀進去記憶體時該怎麼處理？)
- [當訓練的誤差(loss)不再下降時，我要如何終止訓練？](#當訓練的誤差(loss)不再下降時，我要如何終止訓練？)
- [驗證資料集是怎麼從訓練資料中取出的？](#驗證資料集是怎麼從訓練資料中取出的？)
- [資料在訓練的過程中會被隨機打亂嗎？](#資料在訓練的過程中會被隨機打亂嗎？)
- [我如何在每一個訓練的週期，紀錄訓練/測試的誤差和準確率？](#我如何在每一個訓練的週期，紀錄訓練/測試的誤差和準確率？)
- [我如何"凍結"一個 layer](#我如何"凍結"一個 layer)
- [我如何使用一個有狀態的 RNN？](#我如何使用一個有狀態的-RNN？)
- [我如何從循序式模型中移除一層網路？](#我如何從循序式模型中移除一層網路？)
- [如何在 Keras 中使用預先訓練好的模型？](#如何在-Keras-中使用預先訓練好的模型？)
- [如何在 Keras 中使用 HDF5 檔案當成輸入？](#如何在-Keras-中使用-HDF5-檔案當成輸入？)
- [Keras 設定檔案儲存在什麼位置？](#Keras-設定檔案儲存在什麼位置？)
- [在使用 Keras 的開發過程中，我如何得到可以複現的結果？](#在使用-Keras-的開發過程中，我如何得到可以複現的結果？)

---

### 我要如何引用 Keras？

如果 Keras 對你的研究有幫助時，歡迎引用。底下是一個 BibTeX 的範例：

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/fchollet/keras}},
}
```

---

### 我要如何在 GPU 上執行 Keras？

如果你的後端是使用 TensorFlow 或 CNTK 時，只要運作的環境上有 GPU，Keras 就會自動使用它來執行你的程式。

如果你的後端是使用 Theano，可以使用底下幾個方式來進行設定：

方法一：使用 Theano flags

```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

'gpu' 這個命名會隨著你的機器上的名稱而有所變更(例如可能是：`gpu0`、`gpu1` 等)。

方法二：設定`.theanorc` 檔案： [操作方式](http://deeplearning.net/software/theano/library/config.html)

方法三：在你的程式碼中的一開始手動設定 `theano.config.device`、`theano.config.floatX`

```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### "樣本(sample)"、"批次(batch)" 和 "epoch(訓練週期)" 代表什麼意思？

當你想要正確使用 Keras 時，底下是一些你必須要知道的定義：

- **樣本(sample)**: 資料集當中的一筆資料
  - *例如：* 在卷積神經網路 (CNN) 中，一張圖片就是一個**樣本**。
  - *例如：* 在語音辨識模型中，一個音檔就是一個**樣本**。
- **批(batch)**: *N* 個樣本的集合。在一批當中的**樣本**會被獨立且平行化的處理。
  - 一**批**的資料基本上比起單一個樣本而言，可以更好的模擬資料的分佈。一批的資料越多，對於模擬資料分佈的效果越好。然而，執行完一次 batch 也只會更新一次網路的權重。對於評估或預測來說，我們建議你在用光你的記憶體之前，盡可能使用較大的 batch size (通常較大的 batch size 代表進行一次評估或預估的速度較快)。
- **Epoch**: Epoch 通常定義為 "整個神經網路更新完一次整個資料集的參數的過程"。Epoch 會把訓練過程分成數個階段，這樣可以比較好的紀錄和評估模型的訓練過程。
  - 當你在 Keras 中使用 `evaluation_data` 或 `evaluation_split` 的 `fit` 方法時，在每一次 **epoch** 執行完後都會進行一次驗證。
  - 在 Keras 中，你可以在每次 **epoch** 結束後執行 [callbacks](https://keras.io/callbacks/) 函式，用來調整學習率(learning rate) 或是顯示一些模型的資訊。

---

### 我該如何保存 Keras 訓練出來的模型？

#### 儲存/讀取整個模型 (架構 + 權重 + optimizer 狀態)

*我們並不建議使用 pickle 或 cPickle 來保存 Keras 的模型*

你可以使用 `model.save(filepath)` 將 Keras 的模型儲存到一個 HDF5 的檔案中，這當中包含：

- 模型的架構，這可以讓你重建此模型
- 模型的權重
- 訓練的設定 (loss、optimizer)
- optimizer 的狀態，允許你在上次訓練中斷的地方重新開始訓練

你可以使用 `keras.models.load_model(filepath)` 來重新實例化模型。
`load_model` 也會使用儲存的設定來重新編譯你的模型(除非這個模型是第一次使用，之前從來沒有編譯過)。

範例：

```python
from keras.models import load_model

model.save('my_model.h5')  # 建立 HDF5 檔案 'my_model.h5'
del model  # 刪除既有的模型

# 回傳一個編譯後的模型
# 跟前一個相同
model = load_model('my_model.h5')
```

#### 儲存/讀取模型的架構

如果你只需要儲存**模型的架構**，不需要權重和訓練的設定，你可以這樣做：

```python
# 存成 JSON
json_string = model.to_json()

# 存成 YAML
yaml_string = model.to_yaml()
```

產生出來的 JSON/YAML 檔案是可以被讀取的，如果必要時還可以手動修改。

你可以從這些檔案中產生對應的模型：

```python
# 從 JSON 產生模型：
from keras.models import model_from_json
model = model_from_json(json_string)

# 從 YAML 產生模型：
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 僅儲存/讀取模型的權重

如果你要儲存**模型的權重**，你可以使用 HDF5 來達成。

注意，你需要先安裝 HDF5 和 HDF5 的 Python 函式庫，這些是沒有包含在 Keras 當中的。

```python
model.save_weights('my_model_weights.h5')
```

如果你已經初始化你的模型了，你可以透過以下方式讀取參數的權重，要注意你的架構必須*相同*。

```python
model.load_weights('my_model_weights.h5')
```

如果你想要讀取參數到不同的架構時(但某些層是一樣的)，例如像是 fine-tune 或 transfer-learning，你可以指定*層的名稱*來讀取參數：

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：

```python
"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

#### 在已經儲存的模型中處理客製化層 (或其他客製化物件)

如果你想要讀取的模型包含了客製化層、其他客製化的類別或函式，你可以在讀取的時候，透過 `custom_objects` 參數來傳遞他們：

```python
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

或是你也可以使用[客製化物件範圍](https://keras.io/utils/#customobjectscope):

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

在 `load_model`、`model_from_json` 和  `model_from_yaml`這些方法當中處理客製化物件的方式都一樣：

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 為什麼我在訓練階段的誤差(loss)比測試階段來得高？

Keras 的模型包含兩種模式：訓練和測試。而正規化的機制，像是：Dropout、L1/L2 weight 正規化在測試時是被關閉的。

此外，訓練時的誤差是每個 batch 誤差的平均，在訓練的過程中，一般來說，每個 epoch 中的第一個 batch 的誤差會比後面的 batch 要大一些。另一方面，每一個 epoch 結束後所計算的測試誤差是使用模型模型在 epoch 結束時的狀態所決定，這時候的誤差通常比較小一點。

---

### 我要如何得到在訓練時中間層的輸出？

最簡單的做法是建立一個新的 `模型`，讓他輸出你感興趣該中間層的輸出。

```python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

此外，你可以使用 Keras 的函式來得到特定輸入所產生的輸出：

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

同樣的，你也可以直接使用 Theano 和 Tensorflow 的函式。

注意，如果你的模型在訓練和測試階段的行為不同(例如：使用了 `Dropout`、`BatchNormalization`)，你你需要在函式中傳入 learning_phase 這個參數：

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 當資料量太大無法一次讀進去記憶體時該怎麼處理？

你可以用 batch 的訓練方式：`model.train_on_batch(x, y)` 和 `model.test_on_batch(x, y)`。可以參考[模型的文件](/models/sequential)。

此外，你也可以寫一個每次產生一個 batch 樣本的產生器，並呼叫：`model.fit_generator(data_generator, steps_per_epoch, epochs)`。

你可以參考[CIFAR10](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)中的範例來了解這種用法。

---

### 當訓練的誤差(loss)不再下降時，我要如何終止訓練？

你可以使用 `EarlyStopping` 這個 callback 函式：

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

對於 callback，可以參考 [callbacks 文件](/callbacks).

---

### 驗證資料集是怎麼從訓練資料中取出的？

如果你在 `model.fit` 中設定 `validation_split` 參數為，比如說 0.1，那麼驗證資料集會是訓練資料的*最後 10%*。如果你設定為 0.25，那驗證資料集會是最後 25%。注意，在取出驗證資料時，訓練資料並不會被隨機打亂，因此驗證資料就會是*最後的那些資料*。

相同的驗證資料會被用在所有的 epochs 當中。

---

### 資料在訓練的過程中會被隨機打亂嗎？

是的。只要你在 `model.fit` 中，將 `shuffle` 參數設定為 `True` (其實這是預設值)，訓練資料會在每個 epoch 中被隨機打亂。

驗證資料則不會被打亂。

---

### 我如何在每一個訓練的週期，紀錄訓練/測試的誤差和準確率？

`model.fit` 方法會回傳一個 `History` 的 callback，其中有一個屬性是 `history`，當中包含了訓練的誤差和其他評量的結果。

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 我如何"凍結"一個 layer？

"凍結" 一個 layer 代表將它從訓練中移除，換句話說，該 layer 的權重是永遠不會被更新。在進行模型的微調、或使用固定 embedding 在處理文字輸入時，都需要用到這個技巧。 

你可以傳入一個 `trainable` 參數 (boolean) 到 layer 的建構子中來凍結該 layer。

```python
frozen_layer = Dense(32, trainable=False)
```

此外，你可以在模型初始化之後，透過設定 layer 的 `trainable` 參數為 `True` 或 `False` 來決定是否凍結該 layer。如果要讓改變生效，你需要在修改完後呼叫 `compile()` 方法，底下是一個相關範例：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

---

### 我如何使用一個有狀態的 RNN？

Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

建立一個有狀態的 RNN 代表在每個

當你在使用有狀態的 RNN 時，那其實就代表了：

- 所有的 batch 有相同數量的樣本
- If `x1` and `x2` are successive batches of samples, then `x2[i]` is the follow-up sequence to `x1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_size` argument to the first layer in your model. E.g. `batch_size=32` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).
- specify `shuffle=False` when calling fit().

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python

x  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Notes that the methods `predict`, `fit`, `train_on_batch`, `predict_classes`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.

---

### 我如何從循序式模型中移除一層網路？

你可以在循序模型中，透過 `.pop()` 移除最後一個加入的 layer：

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### 如何在 Keras 中使用預先訓練好的模型？

以下關於影像分類的模型已經被訓練好可以直接使用：

- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1

你可以從 `keras.applications` 引用：

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet

model = VGG16(weights='imagenet', include_top=True)
```

關於這些參考範例，可以閱讀 [應用模組文件](/applications) 章節。

關於如何使用這些預先訓練好的模型來做特徵萃取或是更細部的調整，請閱讀[此文章](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)。

VGG16 的模型在許多 Keras 的範例中也是很基本會用到的：

- [Style transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)

---

### 如何在 Keras 中使用 HDF5 檔案當成輸入？

你可以使用 `keras.utils.io_utils` 中的 `HDF5Matrix` 類別。參考 [HDF5Matrix 文件](/utils/#hdf5matrix) 中詳細的說明。

你也可以直接從 HDF5 的檔案格式中讀取資料集：

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

---

### Keras 設定檔案儲存在什麼位置？

Keras 所有的資料預設會儲存在：

```bash
$HOME/.keras/
```

注意，Windows 使用者要將 `$HOME` 改為 `%USERPROFILE%`。

為了預防 Keras 無法使用上述的目錄(比如說因為權限問題)，會使用 `/tmp/.keras` 作為備份目錄。

Keras 的設定檔案是 JSON 格式，儲存在 `$HOME/.keras/keras.json`。預設的設定檔案格式類似如下：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

檔案當中包含以下欄位：

- 在影像處理 layer 或其他工具中所預設的影像資料處理格式 (`channels_last` 或 `channels_first`)。
- 在某些運算中所使用的 `epsilon` 數值模糊因子，防止除以零的情形產生。
- 預設的浮點數資料型態。
- 預設的後端。詳見 [後端文件](/backend)。

Likewise, cached dataset files, such as those downloaded with [`get_file()`](/utils/#get_file), are stored by default in `$HOME/.keras/datasets/`.

同樣的，某些快取的檔案，比如說透過 [`get_file()`](/utils/#get_file) 下載的資料集，預設會被儲存在 `$HOME/.keras/datasets/`。

---

### 在使用 Keras 的開發過程中，我如何得到可以複現的結果？

在開發階段，某些時候得到可以復現的結果是有用的，這樣才能知道性能的改變是因為模型架構的變化，還是因為資料的修改所造成，抑或只是因為隨機取樣的關係。下面的程式碼展示了如何得到可以複現結果的方法，這適用於 Python 3 + 使用 TensorFlow 作為後端的架構：

```python
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
```

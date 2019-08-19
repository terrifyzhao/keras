from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras_bert import load_trained_model_from_checkpoint
from keras.models import Model
from keras.optimizers import Adam

# 因为数据也是从网络上爬取的，所以还需要用正则表达式去除HTML标签
import re
import os


def remove_html(text):
    r = re.compile(r'<[^>]+>')
    return r.sub('', text)


# 观察IMDB文件目录结构，用函数进行读取
def read_file(filetype):
    path = './aclImdb/'
    file_list = []
    positive = path + filetype + '/pos/'
    for f in os.listdir(positive):
        file_list += [positive + f]
    negative = path + filetype + '/neg/'
    for f in os.listdir(negative):
        file_list += [negative + f]
    print('filetype:', filetype, 'file_length:', len(file_list))
    label = ([1] * 12500 + [0] * 12500)  # train数据和test数据中positive都是12500，negative都是12500
    text = []
    for f_ in file_list:
        with open(f_, encoding='utf8') as f:
            text += [remove_html(''.join(f.readlines()))]
    return label, text


# 用x表示label,y表示text里面的内容
x_train, y_train = read_file('train')
x_test, y_test = read_file('test')

token = Tokenizer(num_words=2000)  # 建立一个有2000单词的字典
token.fit_on_texts(y_train)  # 读取所有的训练数据评论，按照单词在评论中出现的次数进行排序，前2000名会列入字典

train_seq = token.texts_to_sequences(y_train)
test_seq = token.texts_to_sequences(y_test)

# 截长补短，让每一个数字列表长度都为100
_train = sequence.pad_sequences(train_seq, maxlen=100)
_test = sequence.pad_sequences(test_seq, maxlen=100)

config_path = 'publish/bert_config.json'
checkpoint_path = 'publish/bert_model.ckpt'
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

x = Input(shape=(None,))
c = Input(shape=(None,))
out = bert_model([x, c])
out = Dense(1, activation='sigmoid')(out)

model = Model([x, c], out)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
model.fit(_train, x_train,
          batch_size=100,
          epochs=10,
          validation_split=0.2)
scores = model.evaluate(_test, x_test)

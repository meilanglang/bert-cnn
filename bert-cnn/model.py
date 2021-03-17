#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,utils,regularizers,optimizers,models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding,Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input,Concatenate,concatenate,Bidirectional,GRU,GlobalMaxPooling1D,GlobalAveragePooling1D,Add,BatchNormalization
from tensorflow.keras import Model,Input
from tensorflow.python.ops import tensor_array_ops
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras.utils import multi_gpu_model
#from tensorflow.keras.attention_keras.layers.attention import AttentionLayer
#K.set_image_data_format('channels_first')
NB_WORDS=87365
EMBEDDING_DIM=768
embedding_matrix=np.load('bert_embedding_matrix.npy')

class AttLayer(Layer):
    def __init__(self, attention_dim,name="AttLayer",**kwargs):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(name=name,**kwargs)
    def get_config(self):
        config = {"attention_dim":self.attention_dim}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)
                  
    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def cnn_model():
    enhancers=Input(shape=(2993,))
    promoters=Input(shape=(1993,))
    # 嵌入层（使用预训练的词向量）
    embed_en=layers.Embedding(NB_WORDS,EMBEDDING_DIM,input_length=2993,weights=[embedding_matrix],trainable=True)(enhancers)
    embed_pr=layers.Embedding(NB_WORDS,EMBEDDING_DIM,input_length=1993,weights=[embedding_matrix],trainable=True)(promoters)
    print(embed_en.shape,embed_pr.shape)
    # 卷积层和池化层，设置卷积核大小分别为3,4,5 
    cnn1_en = Conv1D(128, 3, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_en)
    cnn1_en_pool = GlobalMaxPooling1D()(cnn1_en)
    #ave_pool=GlobalAveragePooling1D()(cnn1_en)
    #cnn1_en_pool=Add()([max_pool,ave_pool])
 
    cnn2_en = Conv1D(128, 4, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_en)
    cnn2_en_pool = GlobalMaxPooling1D()(cnn2_en)
    #ave_pool=GlobalAveragePooling1D()(cnn2_en)
    #cnn2_en_pool=Add()([max_pool,ave_pool])
    
    cnn3_en = Conv1D(128, 5, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_en)
    cnn3_en_pool = GlobalMaxPooling1D()(cnn3_en)
    #ave_pool=GlobalAveragePooling1D()(cnn3_en)
    #cnn3_en_pool=Add()([max_pool,ave_pool])
    
    
    cnn1_pr = Conv1D(128, 3, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_pr)
    cnn1_pr_pool = GlobalMaxPooling1D()(cnn1_pr)
    #ave_pool=GlobalAveragePooling1D()(cnn1_pr)
    #cnn1_pr_pool=Add()([max_pool,ave_pool])
    
    cnn2_pr = Conv1D(128, 4, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_pr)
    cnn2_pr_pool = GlobalMaxPooling1D()(cnn2_pr)
    #ave_pool=GlobalAveragePooling1D()(cnn2_pr)
    #cnn2_pr_pool=Add()([max_pool,ave_pool])
    
    cnn3_pr = Conv1D(128, 5, padding='same', strides=1, activation='relu',kernel_regularizer=l2(1e-4))(embed_pr)
    cnn3_pr_pool = GlobalMaxPooling1D()(cnn3_pr)
    #ave_pool=GlobalAveragePooling1D()(cnn3_pr)
    #cnn3_pr_pool=Add()([max_pool,ave_pool])
    
    # 合并三个模型的输出向量
    cnn_en = concatenate([cnn1_en_pool, cnn2_en_pool, cnn3_en_pool], axis=-1)
    cnn_pr = concatenate([cnn1_pr_pool, cnn2_pr_pool, cnn3_pr_pool], axis=-1)

    merge_layer=Concatenate(axis=1)([cnn_en, cnn_pr])
    print(merge_layer.shape)
    drop = Dropout(0.3)(merge_layer)
    #bn=BatchNormalization()(merge_layer)
    #dense1=Dense(200, activation='relu')(drop)
     #在池化层到全连接层之前可以加上dropout防止过拟合
    preds = Dense(1, activation='sigmoid')(drop)
    
    
    with tf.device('/device:CPU:0'):
        model = Model([enhancers,promoters],preds)
    opt = optimizers.Adam(learning_rate=0.0001)
    model_struct=utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    parallel_model = multi_gpu_model(model, gpus=2,cpu_merge=True,cpu_relocation=True)
    parallel_model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    print(model_struct)
    return parallel_model
import tensorflow as tf
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, LayerNormalization, MultiHeadAttention

# 이미지에서 특징을 추출하는 인코더 클래스
class EncoderCNN(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(EncoderCNN, self).__init__()
        # ResNet101V2 모델을 사용하여 이미지 특징을 추출
        base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        # ResNet101V2의 특정 레이어 출력을 사용하여 특징 추출기 구성
        self.feature_extractor = tf.keras.Model(inputs=base_model.input,
                                                outputs=base_model.get_layer('conv5_block3_out').output)
        self.feature_extractor.trainable = False  # 기본 ResNet101V2 모델의 가중치는 학습되지 않도록 설정
        self.fc = Dense(embedding_dim, activation='relu')  # 추출된 특징을 임베딩 차원으로 변환하는 Dense 레이어

    # 이미지 입력을 받아 특징 벡터로 변환하는 함수
    def call(self, x):
        features = self.feature_extractor(x)  # ResNet101V2를 통해 이미지 특징 추출
        features = self.fc(features)  # 추출된 특징을 Dense 레이어를 통해 임베딩 벡터로 변환ah
        return features

    # 추출된 특징 벡터의 크기를 반환하는 함수
    def get_feature_shape(self):
        return self.feature_extractor.output_shape[1:3]  # 특징 맵의 공간적 크기 반환


# Bahdanau Attention 메커니즘을 구현한 클래스
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)  # 입력 특징에 적용될 Dense 레이어
        self.W2 = Dense(units)  # 숨겨진 상태에 적용될 Dense 레이어
        self.V = Dense(1)  # Attention 점수를 계산하는 데 사용될 Dense 레이어

    # Attention 메커니즘을 적용하는 함수
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # 숨겨진 상태의 차원을 확장하여 덧셈이 가능하게 만듦
        # 특징과 숨겨진 상태를 결합한 후 활성화 함수 tanh를 적용
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attention_hidden_layer)  # Attention 점수를 계산
        attention_weights = tf.nn.softmax(score, axis=1)  # 점수에 softmax를 적용하여 가중치로 변환
        context_vector = attention_weights * features  # 가중치를 적용하여 컨텍스트 벡터 생성
        context_vector = tf.reduce_sum(context_vector, axis=1)  # 컨텍스트 벡터를 합하여 최종 벡터 생성
        return context_vector, attention_weights


# 캡션 생성을 위한 디코더 RNN 클래스
class DecoderRNN(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, num_heads=8):
        super(DecoderRNN, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)  # 단어 임베딩을 위한 레이어
        self.lstm = LSTM(units, return_sequences=True, return_state=True)  # LSTM 레이어
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)  # Multi-Head Attention 레이어
        self.fc1 = Dense(units, activation='relu')  # 특징을 임베딩 차원으로 변환하는 Dense 레이어
        self.fc2 = Dense(vocab_size)  # 최종 출력 단어를 예측하는 Dense 레이어
        self.attention = BahdanauAttention(units)  # Bahdanau Attention 메커니즘
        self.dropout = Dropout(0.5)  # 과적합 방지를 위한 Dropout 레이어
        self.layernorm = LayerNormalization()  # Layer Normalization 레이어

    # 디코더의 순전파 과정을 정의하는 함수
    def call(self, x, features, hidden, cell):
        context_vector, attention_weights = self.attention(features, hidden)  # Attention 메커니즘 적용
        x = self.embedding(x)  # 입력된 단어를 임베딩 벡터로 변환
        # 컨텍스트 벡터와 임베딩 벡터를 결합하여 LSTM에 입력
        lstm_input = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(lstm_input, initial_state=[hidden, cell])  # LSTM을 통해 다음 상태 예측

        # Multi-Head Attention 적용
        attention_output = self.multi_head_attention(output, output, output)  # Self-Attention을 통한 컨텍스트 보강
        attention_output = self.layernorm(attention_output + output)  # 잔차 연결 후 Layer Normalization 적용

        # Fully Connected Layer
        x = self.fc1(attention_output)  # Dense 레이어를 통해 특징 변환
        x = self.dropout(x)  # Dropout 적용
        x = tf.reshape(x, (-1, x.shape[2]))  # 형태 변환
        x = self.fc2(x)  # 최종 출력 단어 예측

        return x, state_h, state_c, attention_weights  # 예측 결과와 상태 반환

    # 디코더 상태를 초기화하는 함수
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))  # 상태를 영벡터로 초기화


# 이미지 캡션 생성을 위한 전체 모델 클래스
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, max_length):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embedding_dim)  # 이미지에서 특징을 추출하는 인코더
        self.attention_features_shape = self.encoder.get_feature_shape()  # 특징 맵의 크기를 저장
        self.decoder = DecoderRNN(embedding_dim, units, vocab_size)  # 특징을 바탕으로 캡션을 생성하는 디코더
        self.max_length = max_length  # 캡션의 최대 길이

    # 모델의 순전파 과정을 정의하는 함수
    def call(self, inputs, training=False):
        img_tensor, target = inputs  # 입력된 이미지와 타겟 캡션
        loss = 0  # 초기 손실을 0으로 설정

        batch_size = tf.shape(img_tensor)[0]  # 배치 크기를 가져옴
        features = self.encoder(img_tensor)  # 이미지를 인코더에 입력하여 특징 추출
        features = tf.reshape(features, (batch_size, -1, features.shape[3]))  # 특징을 적절한 형태로 변환

        dec_hidden, dec_cell = self.decoder.reset_state(batch_size)  # 디코더의 초기 상태를 설정

        all_attention_weights = []  # Attention 가중치를 저장할 리스트

        # 타겟 시퀀스를 순차적으로 처리
        for i in range(1, self.max_length):
            predictions, dec_hidden, dec_cell, attention_weights = self.decoder(
                tf.expand_dims(target[:, i - 1], 1),  # 이전 시간의 단어를 입력
                features, dec_hidden, dec_cell)  # 디코더에 특징과 상태를 입력하여 다음 단어 예측
            loss += self.loss_function(target[:, i], predictions)  # 손실 계산
            all_attention_weights.append(attention_weights)  # Attention 가중치 저장

        # Doubly stochastic attention regularization
        attention_weights = tf.stack(all_attention_weights, axis=1)  # Attention 가중치들을 쌓음
        attention_features = self.attention_features_shape[0] * self.attention_features_shape[1]  # Attention 피처의 크기 계산
        alphas = tf.reduce_sum(1. - tf.reduce_sum(attention_weights, axis=1), axis=-1)  # Attention 정규화 항 계산
        lambda_reg = 1.0  # 정규화 강도 설정
        loss += lambda_reg * tf.reduce_mean(alphas)  # 손실에 정규화 항 추가

        return loss / tf.cast(self.max_length - 1, tf.float32)  # 평균 손실 반환

    # 손실 함수를 정의하는 함수
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # 패딩 토큰을 무시하기 위한 마스크 생성
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)  # 손실 계산
        mask = tf.cast(mask, dtype=loss_.dtype)  # 마스크의 dtype을 손실과 일치시킴
        loss_ *= mask  # 마스크를 손실에 적용

        return tf.reduce_mean(loss_)  # 평균 손실 반환

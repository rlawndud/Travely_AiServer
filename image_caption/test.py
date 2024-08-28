import tensorflow as tf
import os
import pickle
from image_caption.models import ImageCaptioningModel
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog
from googletrans import Translator

# 모델 로드
def model_load():
    # 하이퍼파라미터 설정
    EMBEDDING_DIM = 512
    UNITS = 1024
    MAX_LENGTH = 50

    # 경로 설정
    tokenizer_path = r'C:\Users\WS\Desktop\Pycharm_moum\final_fusion\image_caption\tokenizer.pickle'
    checkpoint_path = r"C:\Users\WS\Desktop\Pycharm_moum\final_fusion\image_caption\checkpoints\train"

    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    vocab_size = len(tokenizer.word_index) + 1

    model = ImageCaptioningModel(EMBEDDING_DIM, UNITS, vocab_size, MAX_LENGTH)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0)
    if os.path.exists(checkpoint_path):
        print("Loading weights from checkpoint...")
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

        if ckpt_manager.latest_checkpoint:
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Checkpoint restored: {ckpt_manager.latest_checkpoint}")
            status.expect_partial()
        else:
            print("No checkpoint found. Using the model with initial weights.")

    return model, tokenizer, MAX_LENGTH

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select an image',
                                           filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.gif')])
    return file_path


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    return tf.expand_dims(tf.cast(img, tf.float32), 0), image_path


# 캡션 디코딩 함수
def decode_caption(sequence, tokenizer):
    index_word = tokenizer.index_word
    return [index_word.get(idx, '<unk>') for idx in sequence if idx != 0]  # Ignore padding (0)


# 이미지 전처리 함수
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return tf.cast(img, tf.float32)  # 명시적으로 float32로 변환


# Greedy Search 방식으로 캡션 생성 함수 (수정됨)
def generate_caption(image_path, model, tokenizer, MAX_LENGTH, greedy=False):
    # 이미지 전처리
    img_tensor = preprocess_image(image_path)
    img_tensor = tf.expand_dims(img_tensor, 0)

    features = model.encoder(img_tensor)
    features = tf.reshape(features, (1, -1, features.shape[3]))

    dec_hidden, dec_cell = model.decoder.reset_state(1)

    start_token = tokenizer.word_index.get('start')
    end_token = tokenizer.word_index.get('end')

    if greedy:
        caption = [start_token]
        for _ in range(MAX_LENGTH):
            predictions, dec_hidden, dec_cell, _ = model.decoder(
                tf.expand_dims(tf.convert_to_tensor([caption[-1]]), 1),
                features, dec_hidden, dec_cell
            )
            predicted_id = tf.argmax(predictions[0]).numpy()
            if predicted_id == end_token:
                break
            caption.append(predicted_id)

    # Convert to readable caption
    caption = decode_caption(caption, tokenizer)

    # Remove 'start' token if it's the first word
    if caption and caption[0].lower() == 'start':
        caption = caption[1:]

    return ' '.join(caption)

# 번역 함수
def translate_to_korean(text):
    translator = Translator()
    translated = translator.translate(text, dest='ko')
    return translated.text


if __name__ == "__main__":
    model, tokenizer, MAX_LENGTH = model_load()
    file_path = select_file()

    english_caption = generate_caption(file_path, model, tokenizer, MAX_LENGTH, greedy=True)
    korean_caption = translate_to_korean(english_caption)

    print("영어 캡션:", english_caption)
    print("한글 캡션:", korean_caption)
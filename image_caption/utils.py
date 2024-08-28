import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import json
import random
import matplotlib.pyplot as plt

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    return tf.cast(img, tf.float32)  # 명시적으로 float32로 변환

def map_func(img_name, cap):
    img_tensor = load_image(img_name)
    return img_tensor, cap

def prepare_dataset(image_paths, captions, tokenizer, max_length, BATCH_SIZE, BUFFER_SIZE):
    def map_func(img_name, cap):
        img_tensor = load_image(img_name)
        cap = cap.numpy().decode('utf-8')
        cap = 'start' + cap + 'end'
        cap_vector = tokenizer.texts_to_sequences([cap])[0]
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences([cap_vector], maxlen=max_length, padding='post', truncating='post')[0]
        return img_tensor, cap_vector

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
    dataset = dataset.map(lambda item1, item2: tf.py_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def create_tokenizer(captions):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=None, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    # 원래 캡션에서 텍스트를 인덱싱하여 단어 인덱스 생성
    tokenizer.fit_on_texts([cap.numpy().decode('utf-8') for cap in captions])

    # 기존 인덱스 가져오기
    original_word_index = tokenizer.word_index

    # 새 인덱스 사전 생성 (0과 1을 비우기 위해 2부터 시작)
    new_word_index = {word: index + 3 for word, index in original_word_index.items()}

    # 0과 1과 2을 비워두기 위해 빈 사전 추가
    new_word_index = {1: 'start', 2: 'end', **new_word_index}

    # 새로운 단어 인덱스를 사용하여 tokenizer의 word_index를 업데이트
    tokenizer.word_index = new_word_index
    tokenizer.index_word = {index: word for word, index in new_word_index.items()}

    return tokenizer

def verify_image_caption_mapping(image_paths, captions, num_samples=5):
    for _ in range(num_samples):
        idx = random.randint(0, len(image_paths) - 1)
        img_path = image_paths[idx]
        cap = captions[idx].numpy().decode('utf-8')

        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Caption: {cap}")
        plt.show()

        print(f"Image path: {img_path}")
        print(f"Caption: {cap}\n")

def load_coco_data(annotation_file, image_folder):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_caption_pairs = {}

    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id not in image_caption_pairs:
            image_caption_pairs[image_id] = []
        image_caption_pairs[image_id].append(caption)

    image_paths = []
    captions = []

    for image_id, caption_list in image_caption_pairs.items():
        image_path = f"{image_folder}/{image_id:012d}.jpg"
        selected_caption = random.choice(caption_list)  # 무작위로 하나의 캡션 선택

        image_paths.append(image_path)
        captions.append(tf.constant(selected_caption))

    # 이미지와 캡션 매핑 확인
    #verify_image_caption_mapping(image_paths, captions)

    return image_paths, captions
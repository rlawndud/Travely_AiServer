import tensorflow as tf
import os
import pickle
from image_caption.models import ImageCaptioningModel
from utils import load_coco_data, create_tokenizer, prepare_dataset
from tqdm import tqdm

# 하이퍼파라미터 설정
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512
MAX_LENGTH = 50
EPOCHS = 15

# Load and prepare data
annotation_file = r'C:\Users\WS\Desktop\coco2017\annotations\captions_train2017.json'
image_folder = r'C:\Users\WS\Desktop\coco2017\train2017'
tokenizer_path = 'tokenizer.pickle'
checkpoint_path = "./checkpoints/train"

image_paths, captions = load_coco_data(annotation_file, image_folder)

# Create or load tokenizer
if os.path.exists(tokenizer_path):
    print("Loading existing tokenizer...")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print("Creating new tokenizer...")
    tokenizer = create_tokenizer(captions)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", len(tokenizer.word_index))

print("\nSample word indices:", list(tokenizer.word_index.items())[:20])
print("\nSample index to word:", list(tokenizer.index_word.items())[:20])

# Split the data
train_size = int(0.1 * len(image_paths))
train_image_paths = image_paths[:train_size]
train_captions = captions[:train_size]
val_image_paths = image_paths[train_size:]
val_captions = captions[train_size:]
val_size = len(val_image_paths) // 128
val_image_paths = val_image_paths[:val_size]
val_captions = val_captions[:val_size]

train_dataset = prepare_dataset(train_image_paths, train_captions, tokenizer, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE)
val_dataset = prepare_dataset(val_image_paths, val_captions, tokenizer, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE)

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2775,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = ImageCaptioningModel(EMBEDDING_DIM, UNITS, vocab_size, MAX_LENGTH)

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
else:
    print("Creating new model...")
    # 체크포인트 설정
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

# 학습 함수 정의
@tf.function
def train_step(img_tensor, target):
    loss = 0
    with tf.GradientTape() as tape:
        loss = model((img_tensor, target), training=True)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

total_batches = sum(1 for _ in train_dataset)

# 학습 루프 수정 (기존 코드의 for 루프 내부)
for epoch in range(EPOCHS):
    total_loss = 0

    with tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as pbar:
        for (batch, (img_tensor, target)) in enumerate(train_dataset):
            batch_loss = train_step(img_tensor, target)
            total_loss += batch_loss

            pbar.set_postfix({'loss': f'{batch_loss.numpy():.4f}'})
            pbar.update(1)

    # 에포크 종료 후 체크포인트 저장
    ckpt_manager.save()

    print(f'Epoch {epoch + 1} Train Loss {total_loss / batch:.6f}')

print("Training completed!")

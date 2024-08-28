import asyncio
import pickle
import os
import logging
from face.models import train, predict as face_predict
from background_cnn.places_cnn import load_places365_model, predict_place
from image_caption.test import generate_caption, translate_to_korean, model_load

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOST = '220.90.180.88'
PORT = 5001
Face_MODEL_DIR = 'face_models'
if not os.path.exists(Face_MODEL_DIR):
    os.makedirs(Face_MODEL_DIR)

model_dict = {}  # 방 인덱스별 모델을 저장할 딕셔너리

# 전역 변수로 모델 로드
background_cnn_model = load_places365_model()
caption_model, tokenizer, MAX_LENGTH = model_load()
logging.info("Background CNN model and Caption model loaded.")

async def handle_client_connection(reader, writer):
    try:
        # 데이터 크기 먼저 수신
        data_size_bytes = await reader.readexactly(4)
        data_size = int.from_bytes(data_size_bytes, byteorder='big')
        logging.info(f"Data size: {data_size} bytes received")

        data = b""
        while len(data) < data_size:
            packet = await reader.read(1000000)
            if not packet:
                break
            data += packet
        logging.info(f"Received data length: {len(data)}")

        received_data = pickle.loads(data)
        room_index = received_data['room_index']

        if 'True' in received_data['create_room']:
            # 새로운 방 생성 시 모델 학습
            logging.info(f"Starting model training for room index {room_index}")
            train_dir = os.path.join(Face_MODEL_DIR, str(room_index))
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

            for member_name, photo in zip(received_data['member_names'], received_data['photos']):
                member_dir = os.path.join(train_dir, member_name)
                if not os.path.exists(member_dir):
                    os.makedirs(member_dir)

            # 회원 사진 저장
            for index, (member_name, photo) in enumerate(zip(received_data['member_names'], received_data['photos'])):
                member_dir = os.path.join(train_dir, member_name)
                photo_path = os.path.join(member_dir, f"{member_name}_{index}.jpg")
                with open(photo_path, 'wb') as f:
                    f.write(photo)
                logging.info(f"Saved photo to: {photo_path}")

            # 모델 학습 및 저장
            model_save_path = os.path.join(Face_MODEL_DIR, f"{room_index}_model.clf")
            knn_clf = train(train_dir, model_save_path=model_save_path)
            model_dict[room_index] = knn_clf
            logging.info(f"Model training completed for room index {room_index}")
        else:
            logging.info('Photo analysis requested')

        if 'True' in received_data['delete_room']:
            train_dir = os.path.join(Face_MODEL_DIR, str(room_index))
            if os.path.exists(train_dir):
                os.remove(train_dir)
            if os.path.exists(train_dir):
                model_save_path = os.path.join(Face_MODEL_DIR, f"{room_index}_model.clf")
                os.remove(model_save_path)

            logging.info(f"Room and Model successfully deleted for room index {room_index}")

        if 'True' in received_data['photo_analyze']:
            face_model_path = os.path.join(Face_MODEL_DIR, f"{room_index}_model.clf")
            if os.path.isfile(face_model_path):
                knn_clf = model_dict.get(room_index)
                if knn_clf is None:
                    with open(face_model_path, 'rb') as f:
                        knn_clf = pickle.load(f)
                        model_dict[room_index] = knn_clf

                # 받은 사진을 파일로 저장
                photo_path = 'temp_photo.jpg'
                with open(photo_path, 'wb') as f:
                    f.write(received_data['photo'])

                # 얼굴 분석 수행
                face_predictions = face_predict(photo_path, knn_clf=knn_clf)

                # Places365 분석 수행
                formatted_background_predictions = predict_place(background_cnn_model, photo_path)

                # Image Caption
                captions = generate_caption(photo_path, caption_model, tokenizer, MAX_LENGTH, greedy=True)
                captions_predictions = translate_to_korean(captions)

                os.remove(photo_path)

                # unknown 예측 제거
                filtered_predictions = [(name, loc) for name, loc in face_predictions if name != "unknown"]

                # 이름 분류
                formatted_face_predictions = "".join([name + "#" for name, _ in filtered_predictions])

                results = {
                    'face_predictions': formatted_face_predictions.strip(),
                    'background_predictions': formatted_background_predictions.strip(),
                    'captions_predictions': captions_predictions
                }
                logging.info(f"Face predictions: {formatted_face_predictions}")
                logging.info(f"Background predictions: {formatted_background_predictions.strip()}")
                logging.info(f"Caption predictions: {captions_predictions}")
                logging.info('Analysis completed and sent successfully')

            else:
                results = {'error': 'Model does not exist'}
                logging.error(f"Model not found for room index {room_index}")
        else:
            results = {'success': 'Room creation prepared'}

        # 결과 반환
        writer.write(pickle.dumps(results))
        await writer.drain()

    except Exception as e:
        logging.error(f"Error in handle_client_connection: {e}", exc_info=True)
        results = {'error': str(e)}
        writer.write(pickle.dumps(results))
        await writer.drain()

    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_client_connection, HOST, PORT)
    addr = server.sockets[0].getsockname()
    logging.info(f"Model server started at {addr}")

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
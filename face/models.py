import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def is_image_file(file_path):
    return '.' in file_path and file_path.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image(file_path):
    if not is_image_file(file_path):
        raise ValueError(f"Unsupported image file format: {file_path}")

    return face_recognition.load_image_file(file_path)

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = []
    y = []

    # 훈련 세트의 각 사람을 순회
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 현재 사람에 대한 각 훈련 이미지를 순회
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image, model='cnn')

            if len(face_bounding_boxes) != 1:
                # 훈련 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                if verbose:
                    print("이미지 {}는 훈련에 적합하지 않음: {}".format(img_path, "얼굴을 찾을 수 없음" if len(face_bounding_boxes) < 1 else "너무 많은 얼굴을 찾음"))
            else:
                # 현재 이미지에 대한 얼굴 인코딩을 훈련 세트에 추가
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if verbose:
        print("총 얼굴 인코딩 수: {}, 라벨 수: {}".format(len(X), len(y)))

    if len(X) == 0 or len(y) == 0:
        raise ValueError("훈련 데이터가 충분하지 않습니다. 유효한 얼굴 인코딩이 없습니다.")

    # KNN 분류기에 가중치를 두기 위해 사용할 이웃 수 결정
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        n_neighbors = max(1, n_neighbors)  # n_neighbors가 최소 1이 되도록 설정
        if verbose:
            print("n_neighbors를 자동으로 선택함:", n_neighbors)

    # KNN 분류기 생성 및 훈련
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 훈련된 KNN 분류기 저장
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("유효하지 않은 이미지 경로: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("knn 분류기를 knn_clf 또는 model_path를 통해 제공해야 합니다")

    # 훈련된 KNN 모델을 로드 (전달된 경우)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 이미지 파일 로드 및 얼굴 위치 찾기
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img, model='cnn')

    # 이미지에서 얼굴을 찾을 수 없는 경우 빈 결과 반환
    if len(X_face_locations) == 0:
        return [("unknown", (0, 0, 0, 0))]  # 인식되지 않은 경우 "unknown" 반환

    # 테스트 이미지에서 얼굴 인코딩 찾기
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # 테스트 얼굴에 대한 최적의 일치를 찾기 위해 KNN 모델 사용
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(closest_distances)

    # 클래스 예측 및 임계값 이내의 분류가 아닌 경우 분류 제거
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Pillow 모듈을 사용하여 얼굴 주위에 상자 그리기
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 얼굴 아래에 이름이 있는 레이블 그리기
        text = str(name)  # name이 numpy 문자열일 경우 일반 문자열로 변환
        text_width, text_height = draw.textbbox((0, 0), text)[2:]
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), text, fill=(255, 255, 255, 255))

    # Pillow 문서에 따라 메모리에서 그리기 라이브러리 제거
    del draw

    # 결과 이미지 표시
    pil_image.show()

if __name__ == "__main__":
    # 모델이 훈련되고 저장되면 다음 번에는 이 단계를 건너뜀.
    print("KNN 분류기 훈련 중...")
    classifier = train(r"C:\Users\WS\Desktop\Pycharm_moum\final_fusion\server_reaction\face_models\1", model_save_path="trained_knn_model.clf", n_neighbors=None)
    print("훈련 완료!")

    # STEP 2: 훈련된 분류기를 사용하여 미지의 이미지에 대한 예측 수행
    for image_file in os.listdir(r"C:\Users\WS\Desktop\face_test"):
        full_file_path = os.path.join(r"C:\Users\WS\Desktop\face_test", image_file)

        print("{}에서 얼굴 찾기".format(image_file))

        # 훈련된 분류기 모델을 사용하여 이미지의 모든 사람 찾기
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 콘솔에 결과 출력
        for name, (top, right, bottom, left) in predictions:
            print("- {}을(를) ({}, {}) 위치에서 찾음".format(name, left, top))

        # 이미지에 결과를 오버레이하여 표시
        show_prediction_labels_on_image(os.path.join(r"C:\Users\WS\Desktop\face_test", image_file), predictions)

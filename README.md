모델 중 image_caption의 학습된 모델의 크기때문에 별도 링크 포함

※ 서버 실행 전, checkpoints를 image_caption폴더에 추가 후 실행 <br>
[checkpoints 폴더 다운로드](https://drive.google.com/drive/folders/13jw46SutpeZ2OvGehJnhfCwJswBMIw30?usp=sharing)

## 실행 방법
### AI Server
1. server_reaction파일의 reaction.py를 PyCharm으로 실행
   해당 파일에서 메인 서버와의 연결을 위해 serverhost와 serverport를 현재의 주소에 맞게 값을 변경
```python
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOST = '220.90.180.88'
PORT = 5001
Face_MODEL_DIR = 'face_models'
if not os.path.exists(Face_MODEL_DIR):
    os.makedirs(Face_MODEL_DIR)
```
개발 환경 : PyCharm 2024.1.1, Pyhton 3.8
## 전체 프로젝트 레포지토리
https://github.com/rlawndud/Travely.git

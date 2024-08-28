import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn.functional as F
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import requests
import os


# Places365 클래스 목록
CLASSES = [
    '공항', '비행기 내부', '공항 터미널', '벽감', '골목길', '원형 극장', '놀이 오락실',
'놀이 공원', '아파트 건물/외부', '수족관', '수로', '아케이드', '아치', '고고학 발굴 현장',
'아카이브', '아레나/하키', '아레나/공연', '아레나/로데오', '군사 기지', '미술관', '미술 학교',
'미술 스튜디오', '예술가의 다락방', '조립 라인', '운동장/외부', '공용 아트리움', '다락방', '강당', '자동차 공장',
'자동차 전시장', '황무지', '베이커리/상점', '발코니/외부', '발코니/내부', '볼풀', '무도회장',
'대나무 숲', '은행 금고', '연회장', '바', '헛간', '헛간문', '야구장', '지하실',
'농구장/실내', '화장실', '실내 시장', '야외 시장', '해변', '해변가 집', '미용실', '침실',
'맥주 정원', '맥주 홀', '침상', '생물학 실험실', '산책로', '배 갑판', '보트하우스', '서점',
'부스/실내', '식물원', '내부 전망창', '볼링장', '권투 링', '다리', '건물 외관',
'투우장', '매장실', '버스 내부', '버스 정류장/실내', '정육점', '버트', '오두막/외부',
'구내 식당', '캠프장', '캠퍼스', '운하/자연', '운하/도시', '캔디 가게', '협곡', '자동차 내부',
'회전목마', '성', '지하묘지', '묘지', '산장', '화학 실험실', '아이의 방', '교회/실내',
'교회/외부', '교실', '클린룸', '절벽', '옷장', '의류 상점', '해안', '조종석', '커피숍',
'컴퓨터실', '컨퍼런스 센터', '회의실', '건설 현장', '옥수수밭', '가축 울타리', '복도',
'오두막', '법원', '안뜰', '시내', '갈라진 틈', '횡단보도', '댐', '델리카트슨', '백화점',
'사막/모래', '사막/식물', '사막 도로', '식당/외부', '식당 홀', '식당', '디스코텍',
'문/외부', '기숙사 방', '도심', '탈의실', '차도', '약국', '엘리베이터 문',
'엘리베이터 로비', '엘리베이터 샤프트', '대사관', '엔진룸', '현관 홀', '에스컬레이터/실내', '발굴 현장', '직물 가게',
'농장', '패스트푸드 레스토랑', '경작된 밭', '야생 초원', '들판 길', '화재 탈출구', '소방서',
'연못', '실내 벼룩시장', '실내 꽃집', '푸드 코트', '축구장', '활엽수림',
'숲길', '산책로', '정원', '분수', '갤리', '차고/실내', '차고/외부',
'주유소', '정자/외부', '실내 잡화점', '외부 잡화점', '기념품 가게', '빙하',
'골프장', '온실/실내', '온실/외부', '동굴', '체육관/실내', '격납고/실내',
'격납고/외부', '항구', '철물점', '건초밭', '헬리포트', '고속도로', '홈오피스', '홈시어터',
'병원', '병실', '온천', '호텔/외부', '호텔 방', '집', '사냥 오두막/외부',
'아이스크림 가게', '빙하', '빙붕', '실내 아이스 스케이팅장', '야외 아이스 스케이팅장', '빙산',
'이글루', '산업 지대', '여인숙/외부', '작은 섬', '실내 자쿠지', '감방', '일본 정원',
'보석 가게', '고물상', '카스바', '개집/외부', '유치원 교실', '주방', '작은 부엌', '습식 실험실',
'석호', '자연 호수', '쓰레기 매립지', '착륙 데크', '세탁소', '잔디밭', '강의실', '입법부 회의실',
'도서관/실내', '도서관/외부', '등대', '거실', '적재 부두', '로비', '수문',
'락커룸', '대저택', '이동식 주택', '실내 시장', '야외 시장', '습지', '무술 체육관', '무덤',
'중세 도시', '중층', '해자/물', '모스크/외부', '모텔', '산', '산길', '눈 덮인 산',
'영화관/실내', '박물관/실내', '박물관/외부', '음악 스튜디오', '자연사 박물관', '유치원',
'양로원', '오두막집', '대양', '사무실', '사무실 건물', '사무실 칸막이', '석유 시추선', '수술실',
'과수원', '오케스트라 피트', '불탑', '궁전', '식료품 저장실', '공원', '실내 주차장', '외부 주차장',
'주차장', '목초지', '테라스', '가든 파빌리온', '애완동물 가게', '약국', '전화 부스', '물리학 실험실',
'피크닉 구역', '부두', '피자 가게', '놀이터', '놀이방', '광장', '연못', '현관', '산책로', '펍/실내',
'경마장', '경주로', '뗏목', '철도 트랙', '열대 우림', '리셉션', '오락실', '수리점',
'주거 지역', '레스토랑', '레스토랑 주방', '레스토랑 테라스', '논밭', '강',
'바위 아치', '옥상 정원', '현수교', '폐허', '활주로', '모래상자', '사우나', '학교 건물', '과학 박물관', '서버실',
'헛간', '구두 가게', '상점 전면', '실내 쇼핑몰', '샤워실', '스키 리조트', '스키 슬로프', '하늘', '고층 건물',
'빈민가', '눈밭', '축구장', '마구간', '야구장', '축구 경기장', '축구장', '실내 무대',
'야외 무대', '계단', '저장실', '거리', '지하철역/승강장', '슈퍼마켓', '스시 바',
'늪지', '수영장/자연', '실내 수영장', '야외 수영장', '야외 유대교 회당', '텔레비전 방',
'텔레비전 스튜디오', '아시아 사원', '왕좌 방', '매표소', '조각 정원', '탑', '장난감 가게',
'기차 내부', '기차역/승강장', '나무 농장', '나무집', '참호', '툰드라', '수중/깊은 바다',
'유틸리티 룸', '계곡', '채소밭', '동물 병원', '고가교', '마을', '포도원', '화산',
'야외 배구장', '대기실', '워터 파크', '물탑', '폭포', '물웅덩이', '파도',
'실내 바', '밀밭', '풍력 발전소', '풍차', '마당', '청소년 호스텔', '젠 정원'
]

def load_places365_model():
    # ResNet50을 기반으로 한 Places365 모델 로드
    model = resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 365)

    # Places365 가중치 다운로드 및 로드
    weights_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    weights_path = 'resnet50_places365.pth.tar'

    if not os.path.exists(weights_path):
        print("가중치 파일 다운로드 중...")
        response = requests.get(weights_url)
        with open(weights_path, 'wb') as f:
            f.write(response.content)

    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((331, 331)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    return img_t.unsqueeze(0)


def predict_place(model, img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)

    # 가장 높은 확률의 인덱스를 찾기
    _, pred = probabilities.topk(1, 1, True, True)
    pred = pred.t()
    return CLASSES[pred[0].item()]

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='이미지 선택',
                                           filetypes=[('이미지 파일', '*.jpg *.jpeg *.png *.bmp *.gif *.jfif')])
    return file_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_places365_model().to(device)

    while True:
        file_path = select_file()
        if not file_path:
            print("파일 선택이 취소되었습니다. 프로그램을 종료합니다.")
            break

        results = predict_place(model, file_path)

        print(f"\n선택된 이미지: {file_path}")
        print("예측 결과:")
        print(f"{results}")

        cont = input("\n다른 이미지를 분석하시겠습니까? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
import numpy as np
import random


def add_single_bad_pixels(image_np, percentage=0.01):
    """
    이미지에 소금-후추 노이즈와 유사한 단일 불량 화소를 추가합니다.
    """
    img = image_np.copy()
    h, w, c = img.shape
    num_bad_pixels = int(h * w * percentage)

    # 불량 화소의 위치를 무작위로 선택
    coords_y = np.random.randint(0, h, size=num_bad_pixels)
    coords_x = np.random.randint(0, w, size=num_bad_pixels)

    # 픽셀 값을 0 (검은색) 또는 255 (흰색)으로 랜덤하게 설정
    for y, x in zip(coords_y, coords_x):
        img[y, x, :] = 0 if random.random() < 0.5 else 255

    return img


def add_cluster_bad_pixels(image_np, percentage=0.005):
    """
    쿼드 베이어 패턴에 맞춰 2x2 크기의 클러스터 불량 화소를 추가합니다.
    클러스터는 항상 짝수 좌표(0, 2, 4...)에서 시작하여 동일 색상 그룹을 모방합니다.
    """
    img = image_np.copy()
    h, w, c = img.shape
    num_bad_pixels_total = int(h * w * percentage)
    num_bad_pixels_generated = 0

    # 클러스터 크기는 2x2 (4픽셀)로 고정
    cluster_pixel_count = 4

    # 이미지 높이나 너비가 2보다 작으면 함수를 실행하지 않음
    if h < 2 or w < 2:
        return img

    while num_bad_pixels_generated < num_bad_pixels_total:
        # 클러스터 시작 위치는 항상 (짝수, 짝수) 좌표가 되도록 랜덤 선택
        # (h - 2) // 2 는 2x2 블록이 들어갈 수 있는 y축 시작점 후보군의 개수
        # 여기에 * 2 를 하여 실제 짝수 좌표를 얻음
        start_y = random.randint(0, (h - 2) // 2) * 2
        start_x = random.randint(0, (w - 2) // 2) * 2

        # 2x2 클러스터 영역을 0 (검은색)으로 마스킹
        img[start_y: start_y + 2, start_x: start_x + 2, :] = 0
        num_bad_pixels_generated += cluster_pixel_count

    return img


def add_column_bad_pixels(image_np, max_bad_columns=2):
    """
    이미지에 세로줄(column) 형태의 불량 화소를 추가합니다.
    """
    img = image_np.copy()
    h, w, c = img.shape
    # 1개 또는 2개의 불량 컬럼을 랜덤하게 생성
    num_bad_columns = random.randint(1, max_bad_columns)

    # 불량 컬럼의 x축 위치를 무작위로 선택 (중복 없이)
    bad_column_indices = random.sample(range(w), num_bad_columns)

    for col_idx in bad_column_indices:
        # 해당 컬럼 전체를 0 (검은색)으로 마스킹
        img[:, col_idx, :] = 0

    return img


def generate_bad_pixels(image_np):
    """
    위 함수들을 모두 호출하여 이미지에 복합적인 불량 화소를 생성합니다.
    클러스터 해결에 집중하기 위해 클러스터의 비율을 상대적으로 높게 설정합니다.
    """
    corrupted_img = image_np.copy()

    # 3가지 유형의 불량 화소를 순차적으로 적용
    # 단일/컬럼 픽셀 비율은 낮추고, 클러스터 픽셀 비율은 높여서 학습 집중도를 조절
    corrupted_img = add_single_bad_pixels(corrupted_img, percentage=0.005)  # 1% -> 0.5%
    corrupted_img = add_cluster_bad_pixels(corrupted_img, percentage=0.01)  # 0.5% -> 1%
    corrupted_img = add_column_bad_pixels(corrupted_img, max_bad_columns=1)  # 최대 2줄 -> 1줄

    return corrupted_img

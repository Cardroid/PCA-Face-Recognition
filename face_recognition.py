import os
from glob import glob
import random
from typing import Dict, List, Tuple, Union
import numpy as np
import math
import cv2

try:
    import matplotlib.pyplot as plt

    IS_MATPLOTLIB_ACTIVE = True
except:
    IS_MATPLOTLIB_ACTIVE = False


SHOW_IMAGES_RANDOM_TABLE_INIT = False


def resize(images: Union[List[np.ndarray], np.ndarray], size: Tuple[int, int]) -> Union[List[np.ndarray], np.ndarray]:
    """영상 데이터 리사이징 함수

    Args:
        images (Union[List[np.ndarray], np.ndarray]): 영상 데이터
        size (Tuple[int, int]): 영상 사이즈

    Returns:
        Union[List[np.ndarray], np.ndarray]: 사이즈가 변환된 영상 데이터
    """
    if isinstance(images, list):
        return [cv2.resize(fdata, size) for fdata in images]
    else:
        return cv2.resize(images, size)


def get_image(path: str) -> Dict[str, List[np.ndarray]]:
    """영상 데이터를 불러오는 함수

    Args:
        path (str): 영상 파일이 포함된 디렉터리 경로

    Returns:
        Dict[str, List[np.ndarray]]: 흑백 영상 데이터 ("train", "test" 레이블이 지정됨)
    """
    labels = ["train", "test"]
    images = {}

    for label in labels:
        # 1. 각 라벨 마다 반복
        # 2. GRAYSCALE(회색조 영상) 모드로 영상을 불러옴
        # 3. 각 라벨이 key, 영상 데이터를 value로 가지는 딕셔너리를 생성
        images[label] = [cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) for fpath in glob(os.path.join(os.path.normpath(path), label, "*.*"))]

    # 불러온 영상을 담은 딕셔너리를 반환
    return images


def image2D_to_array1D(images: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
    """2차원 영상 데이터를 1차원 영상 데이터로 변환하는 함수

    Args:
        images (Union[List[np.ndarray], np.ndarray]): 영상 데이터 (2D)

    Returns:
        Union[List[np.ndarray], np.ndarray]: 영상 데이터 (1D)
    """
    # 2차원 영상을 1차원으로 세워진 배열로 변환하여 반환
    if isinstance(images, list):
        return [fdata.reshape(-1, 1) for fdata in images]
    else:
        return images.reshape(-1, 1)


def array1D_to_image2D(arrays: Union[List[np.ndarray], np.ndarray], size: Tuple[int, int]) -> Union[List[np.ndarray], np.ndarray]:
    """1차원 영상 데이터를 2차원 영상 데이터로 변환하는 함수

    Args:
        arrays (Union[List[np.ndarray], np.ndarray]): 영상 데이터 (1D)
        size (Tuple[int, int]): 영상의 크기

    Returns:
        Union[List[np.ndarray], np.ndarray]: 영상 데이터 (2D)
    """
    # 1차원으로 세워진 배열 영상을 원본 크기(size)의 2차원 영상 데이터로 변환하여 반환
    if isinstance(arrays, list):
        return [fdata.reshape(size[1], size[0]) for fdata in arrays]
    else:
        return arrays.reshape(size[1], size[0])


def show_images(images: Union[List[np.ndarray], np.ndarray], window_name: Union[str, None] = "Images", image_count=16, shuffle_option="table", view_image_size=(600, 750)) -> np.ndarray:
    """영상 데이터를 보여주는 함수

    Args:
        images (Union[List[np.ndarray], np.ndarray]): 영상 데이터 리스트
        window_name (Union[str, None], optional): 윈도우 이름. Defaults to "Images".
        image_count (int, optional): 출력 영상 콜라주 개수. Defaults to 16.
        shuffle_option (str, optional): 영상 섞기 옵션 ("new"=항상 순서대로, "table"=셔플 테이블 생성, else 항상 랜덤). Defaults to "table".
        view_image_size (tuple, optional): 출력 영상 크기. Defaults to (600, 750).

    Returns:
        np.ndarray: 출력 영상
    """
    # 주어진 입력 영상이 하나일 경우
    if isinstance(images, np.ndarray):
        # 만약 영상 사이즈가 출력영상 사이즈(view_image_size)보다 작을 경우, 크기를 키움
        result = (images if images.shape[1] <= view_image_size[0] and images.shape[0] <= view_image_size[1] else cv2.resize(images, view_image_size)).astype(np.uint8)

        # 윈도우 이름이 존재할 경우 윈도우 출력을 사용
        if not window_name is None or window_name != "":
            cv2.imshow(window_name, result)

        # 출력 영상을 반환
        return result

    # 전역 변수 사용
    global SHOW_IMAGES_RANDOM_TABLE_INIT, IMAGE_IDX_TABLE

    # 출력영상이 콜라주의 형태를 취하므로 영상 개수가 부족할 경우, 경고를 출력
    if len(images) < image_count:
        print(f"WARNING: 입력된 이미지의 개수({len(images)}) 보다 출력해야 하는 이미지의 개수({image_count})가 더 많습니다.")
        # 강제로 콜라주 개수를 재계산
        image_count = int(math.sqrt(len(images))) ** 2

    # 입력 영상 하나의 크기
    size_h, size_w = images[0].shape
    # 출력(콜라주) 영상의 한 줄의 개수
    image_count_root = round(math.sqrt(image_count))

    # 출력 영상의 가로 및 세로 길이 계산
    result_size_w = image_count_root * size_w
    result_size_h = image_count_root * size_h

    # 빈 영상 생성
    result = np.zeros((result_size_h, result_size_w), np.uint8)

    # 셔플테이블 생성 과정
    if not SHOW_IMAGES_RANDOM_TABLE_INIT:  # 만약 셔플 테이블이 초기화 되지 않았다면,
        IMAGE_IDX_TABLE = list(range(len(images)))  # 주어진 영상의 개수를 사용해 테이블 초기화
    elif len(IMAGE_IDX_TABLE) != len(images):  # 테이블이 초기화 되었지만, 주어진 영상의 갯수와 다르다면,
        IMAGE_IDX_TABLE = list(range(len(images)))  # 주어진 영상의 개수를 사용해 테이블 초기화
        SHOW_IMAGES_RANDOM_TABLE_INIT = False  # 초기화 완료 flag 설정
    local_image_idx_table = IMAGE_IDX_TABLE  # 현재 사용할 테이블을 선언 및 할당 (전역 테이블에 참조되어 있음)
    if shuffle_option == "new" or (shuffle_option == "table" and not SHOW_IMAGES_RANDOM_TABLE_INIT):  # 셔플옵션(shuffle_option)이 "new"일 경우, 항상 새로운 테이블 생성
        SHOW_IMAGES_RANDOM_TABLE_INIT = True
        random.shuffle(IMAGE_IDX_TABLE)  # 테이블 셔플
    elif shuffle_option == "table":  # 이전 셔플 테이블을 사용한다면,
        if not SHOW_IMAGES_RANDOM_TABLE_INIT:  # 셔플되지 않았을 경우에만 테이블을 셔플
            SHOW_IMAGES_RANDOM_TABLE_INIT = True
            random.shuffle(IMAGE_IDX_TABLE)
    else:  # 아닐경우
        local_image_idx_table = list(range(len(images)))  # 테이블을 순서대로 다시 초기화

    # 결과(콜라주) 영상에 주어진 입력영상을 복사
    image_idx = 0
    for h in range(image_count_root):
        for w in range(image_count_root):
            result[h * size_h : h * size_h + size_h, w * size_w : w * size_w + size_w] = images[local_image_idx_table[image_idx]].astype(np.uint8)
            image_idx += 1

    # 윈도우 이름이 존재할 경우 윈도우 출력을 사용
    if not window_name is None or window_name != "":
        cv2.imshow(window_name, cv2.resize(result, view_image_size))

    # 출력 영상을 반환
    return result


def preprocessing(images: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
    """영상 데이터를 전처리하는 함수

    Args:
        images (Union[List[np.ndarray], np.ndarray]): 전처리가 필요한 영상 데이터

    Returns:
        Union[List[np.ndarray], np.ndarray]: 가우시안 블러가 적용된 영상 데이터
    """
    # 입력 영상을 가우시안 블러로 노이즈를 제거하는 전처리 과정을 거친 후, 반환
    if isinstance(images, list):
        return [cv2.GaussianBlur(fdata, (7, 7), 1) for fdata in images]
    else:
        return cv2.GaussianBlur(images, (7, 7), 1)


def train(
    train_arrays: List[np.ndarray],
    feature_count_ratio: float = 0.03,
    is_use_min_dim: bool = True,
    size: Tuple[int, int] = (120, 150),
    is_debug_view: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """영상 데이터를 학습하여 특징 벡터를 추출하는 함수

    Args:
        train_arrays (List[np.ndarray]): 학습 데이터로 사용할 영상 데이터 (1D)
        feature_count_ratio (float, optional): 추출할 특징 벡터의 개수를 결정하는 비율. Defaults to 0.03.
        is_use_min_dim (bool, optional): 연산량 축소 사용 여부. Defaults to True.
        size (Tuple[int, int], optional): 영상의 크기. Defaults to (120, 150).
        is_debug_view (bool, optional): 디버그 뷰를 출력할지 여부. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 학습 영상의 평균 영상, 변환행렬, 특징값
    """

    # 평균 영상을 구함 (1D)
    avg_train_array: np.ndarray = np.array(train_arrays).sum(0) / len(train_arrays)

    # 디버깅용
    if is_debug_view:
        avg_image = array1D_to_image2D(avg_train_array, size)  # 평균 영상
        show_images(avg_image, "avg_image")

    # 차영상들을 구함 (각 학습 영상 - 평균 영상) (2D)
    minus_arrays = np.array([t_arr - avg_train_array for t_arr in train_arrays])

    # 디버깅용
    if is_debug_view:
        # 차영상 확인
        show_images([m_img - m_img.min() for m_img in array1D_to_image2D(list(minus_arrays), size)], "minus_image")

    # 차영상 배열 생성 (1D)
    minus_arrays = minus_arrays.reshape(len(train_arrays), -1)

    # 사용할 특징 갯수를 구하는 람다식 선언
    feature_count_calc = lambda c, r: int(c * r) if c * r > 1 else 1

    if is_use_min_dim:  # 연산량 축소를 사용할 경우,
        # (차원축소된)공분산 행렬 계산
        covariance_matrix = minus_arrays @ minus_arrays.T
        # 고윳값, 고유백터 계산
        evalue, evector = np.linalg.eig(covariance_matrix)
        # 복소수 타입(complex128)이므로 소수 타입(float32)으로 형변환
        evalue, evector = evalue.astype(np.float32), evector.astype(np.float32)

        # 고윳값 인덱스 딕셔너리 생성
        index_dict = {evalue[i]: i for i in range(len(evalue))}
        # 고윳값을 내림차순으로 정렬
        evalue.sort()
        evalue = evalue[::-1]

        # 고윳값 그래프 확인용
        if IS_MATPLOTLIB_ACTIVE:
            plt.plot(evalue)
            plt.show()

        # 사용하는 고윳값 선별
        evalue_dash = evalue[: feature_count_calc(len(evalue), feature_count_ratio)]
        # 각 고윳값의 index에 맞는 고유백터 선별
        evector_dash = np.array([evector[:, index_dict[ew]] for ew in evalue_dash])

        # 원본 공분산 행렬의 고유백터 계산
        evector_dash = minus_arrays.T @ evector_dash.T
        # 고유백터 정규화
        evector_dash = evector_dash / np.sqrt(np.sum(evector_dash**2))
        evector_dash = evector_dash.T

        # 학습 데이터의 특징값을 계산
        X = evector_dash @ minus_arrays.T
    else:
        # 공분산 행렬 계산
        covariance_matrix = minus_arrays.T @ minus_arrays
        # 고윳값, 고유백터 계산
        evalue, evector = np.linalg.eig(covariance_matrix)
        # 복소수 타입(complex128)이므로 소수 타입(float32)으로 형변환
        evalue, evector = evalue.astype(np.float32), evector.astype(np.float32)

        # 고윳값 인덱스 딕셔너리 생성
        index_dict = {evalue[i]: i for i in range(len(evalue))}
        # 고윳값을 내림차순으로 정렬
        evalue.sort()
        evalue = evalue[::-1]

        # 고윳값 그래프 확인용
        if IS_MATPLOTLIB_ACTIVE:
            plt.plot(evalue)
            plt.show()

        # 사용하는 고윳값 선별
        evalue_dash = evalue[: feature_count_calc(len(evalue), feature_count_ratio)]
        # 각 고윳값의 index에 맞는 고유백터 선별
        evector_dash = np.array([evector[:, index_dict[ew]] for ew in evalue_dash])

        # 디버깅용
        if is_debug_view:
            # 고유백터 확인
            show_images([ev_img - ev_img.min() for ev_img in array1D_to_image2D(list(evector_dash), size)], "evector_dash")

        # 학습 데이터의 특징값을 계산
        X = evector_dash @ minus_arrays.T

    # 학습 영상의 평균 영상, 변환행렬, 특징값 반환
    return avg_train_array, evector_dash, X


def test(
    test_arrays: List[np.ndarray],
    avg_train_array: np.ndarray,
    evector_dash: np.ndarray,
    X: np.ndarray,
    test_data_idx: int,
) -> int:
    """입력된 특징 백터 등의 정보로 학습 영상 목록에서 최대한 일치하는 영상을 찾는 함수

    Args:
        test_arrays (List[np.ndarray]): 테스트 영상 데이터
        avg_train_array (np.ndarray): 학습 영상의 평균 영상
        evector_dash (np.ndarray): 변환행렬
        X (np.ndarray): 학습 영상의 특징값
        test_data_idx (int): 테스트 영상 데이터의 인덱스

    Returns:
        int: 오류율이 가장 낮은 학습 영상의 인덱스
    """
    # 테스트 영상 배열에서 주어진 인덱스의 영상을 가져옴
    test_array = test_arrays[test_data_idx]

    # 평균 영상으로 테스트 영상의 차영상을 계산 (테스트 영상 - 학습 영상의 평균 영상)
    test_minus_arrays = test_array - avg_train_array
    # 테스트 영상의 특징값 계산
    result_matrix = evector_dash @ test_minus_arrays

    # 테스트 영상의 특징값과 각 학습 영상들 간의 유클리드 거리를 계산
    result = {np.sqrt(((result_matrix.T - X[:, t_d]) ** 2).sum()): t_d for t_d in range(X.shape[1])}

    # 테스트 영상과 학습 영상들의 특징간 거리 그래프 출력
    if IS_MATPLOTLIB_ACTIVE:
        plt.plot(result)
        plt.show()

    # 거리가 가장 가까운 학습영상의 index를 가져옴
    __min_value = min(result.keys())
    result_idx = result[__min_value]

    # 결과 index를 반환
    return result_idx


def postprocessing(
    source_test_images: np.ndarray,
    source_train_images: np.ndarray,
    test_data_idx: int,
    result_idx: int,
    min_size: int = 750,
) -> np.ndarray:
    """두 개의 결과 영상을 붙이고 후처리를 수행하는 함수

    Args:
        source_test_images (np.ndarray): 원본 테스트 영상 데이터
        source_train_images (np.ndarray): 원본 학습 영상 데이터
        test_data_idx (int): 테스트 영상 인덱스
        result_idx (int): 결과 영상 인덱스
        min_size (int, optional): 출력 영상의 최소 크기. Defaults to 750.

    Returns:
        np.ndarray: 결과 비교영상
    """
    # 각 index에 알맞는 원본 영상을 가져옴
    test_image = source_test_images[test_data_idx]
    target_image = source_train_images[result_idx]

    # 두 영상을 가로로 이어붙인 크기의 빈 영상을 생성
    result_compare_image = np.zeros((test_image.shape[0] if test_image.shape[0] > target_image.shape[0] else target_image.shape[0], test_image.shape[1] + target_image.shape[1]), np.uint8)

    # 결과 비교영상에 두 영상(테스트 영상, 거리가 가장 가까운 학습 영상)을 복사
    result_compare_image[: test_image.shape[0], : test_image.shape[1]] = test_image
    result_compare_image[: target_image.shape[0], test_image.shape[1] :] = target_image

    # 결과 비교영상이 너무 작을 경우 사이즈 조절
    result_compare_image_wh_ratio = result_compare_image.shape[0] / result_compare_image.shape[1]
    if result_compare_image.shape[1] < min_size or result_compare_image.shape[0] < min_size * result_compare_image_wh_ratio:
        result_compare_image = cv2.resize(result_compare_image, (min_size, int(min_size * result_compare_image_wh_ratio)))

    # 결과 비교영상에 정보 추가
    result_compare_image = cv2.putText(
        result_compare_image, f"(Left) TEST index: {test_data_idx}", (5, result_compare_image.shape[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA
    )
    result_compare_image = cv2.putText(
        result_compare_image, f"(Right) TARGET index: {result_idx}", (5, result_compare_image.shape[0] - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA
    )

    # 결과 비교영상을 반환
    return result_compare_image


def app():
    global IS_MATPLOTLIB_ACTIVE

    ########
    # 옵션 #
    ########

    is_plot_view = False  # 그래프 보기 (matplotlib 패키지 필요)
    is_debug_view = True  # 디버그 정보 보기
    is_param_debug = False  # 영상이 아닌 정해진 디버깅용 수치를 사용 (연산 테스트용)
    is_use_min_dim = True  # 연산량 축소 옵션 사용
    is_preprocessing = True  # 영상 전처리 사용
    feature_count_ratio = 0.03  # 사용할 고윳값 개수의 비율
    datasets_dirpath = "face_img"  # 데이터 셋 디렉터리 위치
    size = (120, 150)  # 영상 크기 조절 (가로, 세로)

    ##########
    # 초기화 #
    ##########

    IS_MATPLOTLIB_ACTIVE = IS_MATPLOTLIB_ACTIVE and is_plot_view
    datasets_dirpath = os.path.normpath(datasets_dirpath)

    if is_param_debug:
        size = (3, 3)
        train_images = source_train_images = [
            np.array([[225, 229, 48], [251, 33, 238], [0, 255, 217]], np.float32),
            np.array([[10, 219, 24], [255, 18, 247], [17, 255, 2]], np.float32),
            np.array([[196, 35, 234], [232, 59, 244], [243, 57, 226]], np.float32),
            np.array([[255, 223, 224], [255, 0, 255], [249, 255, 235]], np.float32),
        ]
        test_images = source_test_images = [np.array([[20, 244, 44], [246, 21, 244], [4, 255, 2]], np.float32)]
    else:
        # 영상 로드
        images = get_image(datasets_dirpath)
        # 학습 영상들과 테스트 영상을 각각 분리
        train_images, test_images = source_train_images, source_test_images = images["train"], images["test"]
        print(f"학습 영상 개수: {len(train_images)}")
        print(f"테스트 영상 개수: {len(test_images)}")
        if is_preprocessing:  # 전처리 젹용
            train_images, test_images = preprocessing(train_images), preprocessing(test_images)

        # 리사이징 적용
        train_images, test_images = resize(train_images, size), resize(test_images, size)

    # 2차원 영상을 1차원 배열으로 변환
    train_arrays, test_arrays = image2D_to_array1D(train_images), image2D_to_array1D(test_images)

    while True:
        try:
            test_data_idx = int(input("테스트 영상의 index를 입력해주세요: "))  # 테스트 데이터 Index
            if test_data_idx < 0 or len(test_images) <= test_data_idx:
                print("index 값이 테스트 영상의 개수를 초과하거나 0보다 작습니다. 다시 시도해 주세요.")
            else:
                break
        except:
            print("파싱 할 수 없습니다. 다시 시도해 주세요.")

    if is_debug_view:
        show_images(train_images, "train_images")

    ####################
    # 학습 (특징 추출) #
    ####################

    avg_train_array, evector_dash, X = train(train_arrays, feature_count_ratio, is_use_min_dim, size, is_debug_view)

    ##########
    # 테스트 #
    ##########

    result_idx = test(test_arrays, avg_train_array, evector_dash, X, test_data_idx)

    ##########
    # 후처리 #
    ##########

    result_compare_image = postprocessing(source_test_images, source_train_images, test_data_idx, result_idx)

    #############
    # 결과 확인 #
    #############

    cv2.imshow("Result Compare Image", cv2.resize(result_compare_image, (result_compare_image.shape[1], result_compare_image.shape[0])))

    cv2.waitKey(0)


if __name__ == "__main__":
    app()

from glob import glob
import os
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


def preprocessing(images: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
    if isinstance(images, list):
        return [cv2.GaussianBlur(fdata, (7, 7), 1) for fdata in images]
    else:
        return cv2.GaussianBlur(images, (7, 7), 1)


def resize(images: Union[List[np.ndarray], np.ndarray], size: Tuple[int, int]) -> Union[List[np.ndarray], np.ndarray]:
    if isinstance(images, list):
        return [cv2.resize(fdata, size) for fdata in images]
    else:
        return cv2.resize(images, size)


def get_image(path: str) -> Dict[str, List[np.ndarray]]:
    labels = ["train", "test"]
    images = {}

    for label in labels:
        images[label] = [cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) for fpath in glob(os.path.join(os.path.normpath(path), label, "*.*"))]

    return images


def show_images(images: Union[List[np.ndarray], np.ndarray], window_name: Union[str, None] = "Images", image_count=16, shuffle_option="table", view_image_size=(600, 750)) -> np.ndarray:
    if isinstance(images, np.ndarray):
        result = (images if images.shape[1] <= view_image_size[0] and images.shape[0] <= view_image_size[1] else cv2.resize(images, view_image_size)).astype(np.uint8)

        if not window_name is None or window_name != "":
            cv2.imshow(window_name, result)

        return result

    global SHOW_IMAGES_RANDOM_TABLE_INIT, IMAGE_IDX_TABLE

    if len(images) < image_count:
        print(f"WARNING: 입력된 이미지의 개수({len(images)}) 보다 출력해야 하는 이미지의 개수({image_count})가 더 많습니다.")
        image_count = int(math.sqrt(len(images))) ** 2

    size_h, size_w = images[0].shape
    image_count_root = round(math.sqrt(image_count))

    result_size_w = image_count_root * size_w
    result_size_h = image_count_root * size_h

    result = np.zeros((result_size_h, result_size_w), np.uint8)

    if not SHOW_IMAGES_RANDOM_TABLE_INIT:
        IMAGE_IDX_TABLE = list(range(len(images)))
    elif len(IMAGE_IDX_TABLE) != len(images):
        IMAGE_IDX_TABLE = list(range(len(images)))
        SHOW_IMAGES_RANDOM_TABLE_INIT = False
    local_image_idx_table = IMAGE_IDX_TABLE
    if shuffle_option == "new" or (shuffle_option == "table" and not SHOW_IMAGES_RANDOM_TABLE_INIT):  # 항상 새로운 테이블 생성
        SHOW_IMAGES_RANDOM_TABLE_INIT = True
        random.shuffle(IMAGE_IDX_TABLE)
    elif shuffle_option == "table":
        if not SHOW_IMAGES_RANDOM_TABLE_INIT:
            SHOW_IMAGES_RANDOM_TABLE_INIT = True
            random.shuffle(IMAGE_IDX_TABLE)
    else:
        local_image_idx_table = list(range(len(images)))

    image_idx = 0
    for h in range(image_count_root):
        for w in range(image_count_root):
            result[h * size_h : h * size_h + size_h, w * size_w : w * size_w + size_w] = images[local_image_idx_table[image_idx]].astype(np.uint8)
            image_idx += 1

    if not window_name is None or window_name != "":
        cv2.imshow(window_name, cv2.resize(result, view_image_size))

    return result


def image2D_to_array1D(images: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
    if isinstance(images, list):
        return [fdata.reshape(-1, 1) for fdata in images]
    else:
        return images.reshape(-1, 1)


def array1D_to_image2D(arrays: Union[List[np.ndarray], np.ndarray], size: Tuple[int, int]) -> Union[List[np.ndarray], np.ndarray]:
    if isinstance(arrays, list):
        return [fdata.reshape(size[1], size[0]) for fdata in arrays]
    else:
        return arrays.reshape(size[1], size[0])


def app():
    global IS_MATPLOTLIB_ACTIVE

    ########
    # 옵션 #
    ########
    is_plot_view = False  # 그래프 보기
    is_debug_view = False  # 디버그 정보 보기
    is_param_debug = False  # 영상이 아닌 정해진 디버깅용 수치를 사용 (연산 테스트용)
    is_use_min_dim = True  # 연산량 축소 옵션 사용
    is_preprocessing = True  # 영상 전처리 사용
    feature_count = 3  # 사용할 고윳값 개수
    test_data_idx = 0  # 테스트 데이터 Index
    datasets_dirpath = "src/homework/face_img"  # 데이터 셋 디렉터리 위치
    size = (120, 150)  # 영상 크기 조절 (가로, 세로)

    ##########
    # 초기화 #
    ##########

    IS_MATPLOTLIB_ACTIVE = IS_MATPLOTLIB_ACTIVE and is_plot_view
    datasets_dirpath = os.path.normpath(datasets_dirpath)

    if is_param_debug:
        size = (3, 3)
        train_images = [
            np.array([[225, 229, 48], [251, 33, 238], [0, 255, 217]], np.float32),
            np.array([[10, 219, 24], [255, 18, 247], [17, 255, 2]], np.float32),
            np.array([[196, 35, 234], [232, 59, 244], [243, 57, 226]], np.float32),
            np.array([[255, 223, 224], [255, 0, 255], [249, 255, 235]], np.float32),
        ]
        test_images = [np.array([[20, 244, 44], [246, 21, 244], [4, 255, 2]], np.float32)]
    else:
        images = get_image(datasets_dirpath)
        train_images, test_images = source_train_images, source_test_images = images["train"], images["test"]
        if is_preprocessing:
            train_images, test_images = preprocessing(train_images), preprocessing(test_images)
        train_images, test_images = resize(train_images, size), resize(test_images, size)

    train_arrays, test_arrays = image2D_to_array1D(train_images), image2D_to_array1D(test_images)

    if is_debug_view:
        show_images(train_images, "train_images")

    ####################
    # 학습 (특징 추출) #
    ####################

    avg_train_array: np.ndarray = np.array(train_arrays).sum(0) / len(train_arrays)

    if is_debug_view:
        avg_image = array1D_to_image2D(avg_train_array, size)  # 평균 영상
        show_images(avg_image, "avg_image")

    minus_arrays = np.array([t_arr - avg_train_array for t_arr in train_arrays])

    if is_debug_view:
        show_images([m_img - m_img.min() for m_img in array1D_to_image2D(list(minus_arrays), size)], "minus_image")

    minus_arrays = minus_arrays.reshape(len(train_images), -1)

    if is_use_min_dim:  # 연산량 축소
        covariance_matrix = minus_arrays @ minus_arrays.T
        evalue, evector = np.linalg.eig(covariance_matrix)
        evalue, evector = evalue.astype(np.float32), evector.astype(np.float32)

        index_dict = {evalue[i]: i for i in range(len(evalue))}
        evalue.sort()
        evalue = evalue[::-1]

        if IS_MATPLOTLIB_ACTIVE:
            plt.plot(evalue)
            plt.show()

        evalue_dash = evalue[:feature_count]
        evector_dash = np.array([evector[:, index_dict[ew]] for ew in evalue_dash])

        evector_dash = minus_arrays.T @ evector_dash.T
        evector_dash = evector_dash / np.sqrt(np.sum(evector_dash**2))
        evector_dash = evector_dash.T

        X = evector_dash @ minus_arrays.T
    else:
        covariance_matrix = minus_arrays.T @ minus_arrays
        evalue, evector = np.linalg.eig(covariance_matrix)
        evalue, evector = evalue.astype(np.float32), evector.astype(np.float32)

        index_dict = {evalue[i]: i for i in range(len(evalue))}
        evalue.sort()
        evalue = evalue[::-1]

        if IS_MATPLOTLIB_ACTIVE:
            plt.plot(evalue)
            plt.show()

        evalue_dash = evalue[:feature_count]
        evector_dash = np.array([evector[:, index_dict[ew]] for ew in evalue_dash])

        if is_debug_view:
            show_images([ev_img - ev_img.min() for ev_img in array1D_to_image2D(list(evector_dash), size)], "evector_dash")

        X = evector_dash @ minus_arrays.T

    ##########
    # 테스트 #
    ##########

    test_array = test_arrays[test_data_idx]

    test_minus_arrays = test_array - avg_train_array
    result_matrix = evector_dash @ test_minus_arrays

    result = []
    for t_d in range(X.shape[1]):
        result.append(np.sqrt(((result_matrix.T - X[:, t_d]) ** 2).sum()))

    if IS_MATPLOTLIB_ACTIVE:
        plt.plot(result)
        plt.show()

    result_idx = 0
    __min_value = result[0]
    for i, m in enumerate(result, 1):
        if m < __min_value:
            __min_value = m
            result_idx = i

    test_image = source_test_images[test_data_idx]
    target_image = source_train_images[result_idx]

    result_compare_image = np.zeros((test_image.shape[0] if test_image.shape[0] > target_image.shape[0] else target_image.shape[0], test_image.shape[1] + target_image.shape[1]), np.uint8)

    result_compare_image[: test_image.shape[0], : test_image.shape[1]] = test_image
    result_compare_image[: target_image.shape[0], test_image.shape[1] :] = target_image

    #############
    # 결과 확인 #
    #############

    cv2.imshow("Result Compare Image", cv2.resize(result_compare_image, (result_compare_image.shape[1], result_compare_image.shape[0])))

    cv2.waitKey(0)


if __name__ == "__main__":
    app()

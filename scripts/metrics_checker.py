import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class ArtQualityEvaluator:

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Сдаем заготовки под LPIPS и DINOv2
        # self.lpips = lpips.LPIPS(net='vgg').to(device)

    def compute_edge_sharpness_penalty(self, img_np: np.ndarray) -> float:
        """Оценка избыточной резкости (градиенты Лапласа).

        Возвращает высокий балл, если присутствуют резкие 'цифровые' границы.
        """
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)

    def compute_color_saturation_shift(
        self, img_orig_np: np.ndarray, img_up_np: np.ndarray
    ) -> float:
        """Проверка ухода в 'кислотные' цвета относительно исходника."""
        hsv_orig = cv2.cvtColor(img_orig_np, cv2.COLOR_RGB2HSV)
        hsv_up = cv2.cvtColor(img_up_np, cv2.COLOR_RGB2HSV)

        # Сравниваем среднее значение Saturation (канал S)
        sat_orig = hsv_orig[:, :, 1].mean()
        sat_up = hsv_up[:, :, 1].mean()

        # Возвращает прирост насыщенности
        return float(max(0, sat_up - sat_orig))

    def evaluate_pair(
        self, orig_lr_path: str, generated_path: str
    ) -> dict[str, float]:
        orig = cv2.imread(orig_lr_path)
        gen = cv2.imread(generated_path)
        gen_resized_back = cv2.resize(gen, (orig.shape[1], orig.shape[0]))

        sharpness = self.compute_edge_sharpness_penalty(gen)
        sat_shift = self.compute_color_saturation_shift(orig, gen_resized_back)

        return {
            "sharpness_score": sharpness,
            "saturation_shift": sat_shift,
            # Дополнительно сюда добавятся вызовы DINOv2 и Aesthetic predictor
        }

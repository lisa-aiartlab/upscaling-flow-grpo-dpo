import json
from pathlib import Path
import torch
from PIL import Image


class BaselineReferenceLogger:

    def __init__(self, output_dir: str = "./p_ref_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        self.manifest = []

    @torch.no_grad()
    def process_and_save(self, model, lr_image_path: str, prompt: str, idx: int):
        """Прогон $x_{lr}$ через $p_{ref}$ модель с сохранением результатов и метаданных."""
        lr_img = Image.open(lr_image_path).convert("RGB")

        # Эмуляция пайплайна генерации (заменится на реальный ControlNet/Flow Matching pipeline)
        # generated_img = model(lr_img, prompt)
        generated_img = lr_img.resize(
            (lr_img.width * 4, lr_img.height * 4)
        )  # временная заглушка

        img_filename = f"pref_{idx:05d}.png"
        save_path = self.output_dir / "images" / img_filename
        generated_img.save(save_path)

        record = {
            "id": idx,
            "lr_path": str(lr_image_path),
            "pref_generated_path": str(save_path),
            "prompt": prompt,
        }
        self.manifest.append(record)

    def finalize(self):
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        print(
            f"Базовый датасет p_ref сохранен в {self.output_dir}. Всего кадров: {len(self.manifest)}"
        )

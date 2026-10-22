# ========================================
# Image Helpers: Media Contracts
# ========================================
from pathlib import Path
import tempfile

from PIL import Image
import data_hub as dh

with tempfile.TemporaryDirectory() as tmp:
    image_path = Path(tmp) / "sample.png"
    Image.new("RGB", size=(80, 40), color=(40, 120, 220)).save(image_path)

    pil_image = dh.load_image(image_path)
    tensor = dh.image_to_tensor(image_path, target_size=(32, 64))
    array = dh.image_to_numpy(image_path, target_size=(32, 64))

    print("PIL mode/size:", pil_image.mode, pil_image.size)
    print("tensor shape/dtype:", tuple(tensor.shape), tensor.dtype)
    print("NumPy shape/dtype:", array.shape, array.dtype)

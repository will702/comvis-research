import os
import cv2
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR, CLASSES


def apply_augmentation(img):
    """
    Applies a randomised set of transformations.
    Each call produces a different result so repeated calls on the same
    source image yield genuinely different training samples.
    """
    # 1. Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Random rotation (-25°..+25°) + zoom (0.8–1.2)
    angle = random.uniform(-25, 25)
    scale = random.uniform(0.80, 1.20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Color jitter — brightness & saturation
    if random.random() > 0.3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.75, 1.25)   # saturation
        hsv[:, :, 2] *= random.uniform(0.75, 1.25)   # brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Gamma correction (simulates camera exposure variation)
    if random.random() > 0.4:
        gamma = random.uniform(0.7, 1.5)
        lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                        for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    # 5. Random contrast adjustment
    if random.random() > 0.4:
        alpha = random.uniform(0.8, 1.3)   # contrast factor
        beta  = random.randint(-20, 20)    # brightness offset
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 6. Gaussian noise (simulates sensor noise / image compression)
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 7. Gaussian blur (occasional, mild)
    if random.random() > 0.7:
        ksize = random.choice([(3, 3), (5, 5)])
        img = cv2.GaussianBlur(img, ksize, 0)

    # 8. Unsharp mask / sharpen (occasional)
    if random.random() > 0.7:
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def split_and_augment(test_size=0.2, random_state=42, target_count=300):
    """
    Splits the dataset into train / test (stratified), then:
      - Downsamples majority classes to `target_count`
      - Oversamples minority classes to `target_count` using augmentation

    The test set is NEVER augmented or modified — it reflects the true
    class distribution of the raw data.

    Parameters
    ----------
    target_count : int
        Number of training images per class after balancing.
        Lower value = more aggressive downsampling of majority classes
        and fewer (but richer) augmented minority-class samples.
    """
    print(f"Preparing split and augmentation (target={target_count} per class)...")
    base_split_dir = os.path.join(os.path.dirname(DATA_DIR), 'data_split')
    train_dir = os.path.join(base_split_dir, 'train')
    test_dir  = os.path.join(base_split_dir, 'test')

    # Fresh directories
    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        for cls in CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)

    # Collect all source images
    data = []
    for cls_name in CLASSES:
        cls_path = os.path.join(DATA_DIR, cls_name)
        if not os.path.exists(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.endswith('.jpg'):
                data.append((os.path.join(cls_path, f), cls_name))

    if not data:
        print("No images found!")
        return

    # Stratified train/test split on original data (no augmentation involved)
    paths, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Copy test set as-is (untouched)
    print(f"Copying {len(X_test)} images to test set (untouched)...")
    for path, cls in zip(X_test, y_test):
        shutil.copy(path, os.path.join(test_dir, cls, os.path.basename(path)))

    # Organise train paths by class
    train_classes = {cls: [] for cls in CLASSES}
    for path, cls in zip(X_train, y_train):
        train_classes[cls].append(path)

    counts = {cls: len(train_classes[cls]) for cls in CLASSES}
    print(f"Original train counts : {counts}")
    print(f"Target per class      : {target_count}")

    for cls in CLASSES:
        cls_paths   = train_classes[cls]
        target_path = os.path.join(train_dir, cls)
        current     = len(cls_paths)

        if current >= target_count:
            # Majority class — downsample
            print(f"  [{cls}] Downsampling {current} → {target_count}")
            for p in random.sample(cls_paths, target_count):
                shutil.copy(p, os.path.join(target_path, os.path.basename(p)))
        else:
            # Minority class — copy originals then augment to fill
            print(f"  [{cls}] Oversampling {current} → {target_count} "
                  f"(+{target_count - current} augmented)")
            for p in cls_paths:
                shutil.copy(p, os.path.join(target_path, os.path.basename(p)))

            needed = target_count - current
            for i in range(needed):
                src = random.choice(cls_paths)
                img = cv2.imread(src)
                if img is None:
                    continue
                aug_img = apply_augmentation(img)
                cv2.imwrite(
                    os.path.join(target_path, f"aug_{i}_{os.path.basename(src)}"),
                    aug_img,
                )

    print("Data split and augmentation complete!")
    # Summary
    for split, d in [('Train', train_dir), ('Test', test_dir)]:
        print(f"\n{split} set:")
        for cls in CLASSES:
            n = len([f for f in os.listdir(os.path.join(d, cls)) if f.endswith('.jpg')])
            print(f"  {cls}: {n}")


if __name__ == "__main__":
    split_and_augment(target_count=300)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'acne_1024')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

CLASSES = ['acne1_1024', 'acne2_1024', 'acne3_1024']
IMG_SIZE = (512, 512)

# Ensure directories exist
for d in [MODELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

import os

# adjust your paths here
BASE_PATH = '/home/kathanal/ForDigitHealth/data/'

TARGET_EMOTION_DIM = 'arousal'
FEATURE = 'BPM'
PATH_TO_FEATURE = os.path.join(BASE_PATH, f"feature_segments/{FEATURE}")
PATH_TO_LABELS = os.path.join(BASE_PATH, 'label_segments')
PATH_TO_METADATA = os.path.join(BASE_PATH, 'metadata')

PARTITION_FILES = os.path.join(BASE_PATH, 'metadata/partition.csv')
OUTPUT_PATH = '/home/kathanal/ForDigitHealth/results/'

WIN_LEN = 300
HOP_LEN = 50

LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction')

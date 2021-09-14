import os

# adjust your paths here
BASE_PATH = '/home/kathanal/ForDigitHealth/data/'

PATH_TO_ALIGNED_FEATURES = {
    'stress': os.path.join(BASE_PATH, 'feature_segments')
}

PATH_TO_LABELS = {
    'stress': os.path.join(BASE_PATH, 'label_segments')
}

PATH_TO_METADATA = {
    'stress': os.path.join(BASE_PATH, 'metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

OUTPUT_PATH = '/home/kathanal/ForDigitHealth/results/'
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction')

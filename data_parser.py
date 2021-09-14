import os
import numpy as np
import pandas as pd
import pickle
import config


def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:
        vid, partition = str(row[0]), row[-1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)

    return vid2partition, partition2vid


def segment_sample(sample, win_len, hop_len, segment_type='normal'):
    segmented_sample = []
    assert hop_len <= win_len and win_len >= 10

    segment_ids = sorted(set(sample['segment_id'].values))
    if segment_type in ['by_segs', 'by_segs_only']:
        for id in segment_ids:
            segment = sample[sample['segment_id'] == id]
            if segment_type == 'by_segs_only':
                segmented_sample.append(segment)
            else:
                for s_idx in range(0, len(segment), hop_len):
                    e_idx = min(s_idx + win_len, len(segment))
                    sub_segment = segment.iloc[s_idx:e_idx]
                    segmented_sample.append(sub_segment)
                    if e_idx == len(segment):
                        break
    elif segment_type == 'normal':
        for s_idx in range(0, len(sample), hop_len):
            e_idx = min(s_idx + win_len, len(sample))
            segment = sample.iloc[s_idx:e_idx]
            segmented_sample.append(segment)
            if e_idx == len(sample):
                break
    else:
        print('No such segmentation available.')
    return segmented_sample


def normalize_data(data, idx_list, column_name='feature'):
    train_data = np.row_stack(data['train'][column_name])
    train_mean = np.nanmean(train_data, axis=0)
    train_std = np.nanstd(train_data, axis=0)

    for partition in data.keys():
        for i in range(len(data[partition][column_name])):
            for s_idx, e_idx in idx_list:
                data[partition][column_name][i][:, s_idx:e_idx] = \
                    (data[partition][column_name][i][:, s_idx:e_idx] - train_mean[s_idx:e_idx]) / (
                            train_std[s_idx:e_idx] + 1e-6)  # standardize
                data[partition][column_name][i][:, s_idx:e_idx] = np.where(  # replace any nans with zeros
                    np.isnan(data[partition][column_name][i][:, s_idx:e_idx]), 0.0,
                    data[partition][column_name][i][:, s_idx:e_idx])

    return data


def load_data(normalize=True, norm_opts=None, save=False, apply_segmentation=True):
    feature_path = config.PATH_TO_FEATURE
    label_path = config.PATH_TO_LABELS

    data_file_name = f'data{"_".join(config.FEATURE)}_{config.TARGET_EMOTION_DIM}_{"norm_" if normalize else ""}{config.WIN_LEN}_' \
        f'{config.HOP_LEN}{"_seg" if apply_segmentation else ""}.pkl'
    data_file = os.path.join(f"{config.OUTPUT_PATH}data/", data_file_name)

    if os.path.exists(data_file):  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(config.PARTITION_FILES)

    feature_idx = 2  # first to columns are timestamp and segment_id, features start with the third column

    for partition, vids in partition2vid.items():
        for vid in vids:
            sample_data = []

            # parse features
            feature_file = os.path.join(config.PATH_TO_FEATURE, vid + '.csv')
            assert os.path.exists(
                feature_file), f'Error: no available "{config.FEATURE}" feature file for video "{vid}": "{feature_file}".'
            df = pd.read_csv(feature_file)
            sample_data.append(df)

            # parse labels
            label_file = os.path.join(label_path, config.TARGET_EMOTION_DIM, vid + '.csv')
            assert os.path.exists(
                label_file), f'Error: no available "{config.TARGET_EMOTION_DIM}" label file for video "{vid}": "{label_file}".'
            df = pd.read_csv(label_file)

            label_data = pd.DataFrame(data=df['value'].values, columns=[config.TARGET_EMOTION_DIM])
            sample_data.append(label_data)

            # concat
            sample_data = pd.concat(sample_data, axis=1)
            if partition != 'test':
                sample_data = sample_data.dropna()

            # segment
            if apply_segmentation:
                if partition == 'train':
                        samples = segment_sample(sample_data, config.WIN_LEN, config.HOP_LEN, 'normal')
                else:
                    samples = [sample_data]
            else:
                samples = [sample_data]

            # store
            for i, segment in enumerate(samples):  # each segment has columns: timestamp, segment_id, features, labels
                n_emo_dims = 1
                if len(segment.iloc[:, feature_idx:-n_emo_dims].values) > 0:  # check if there are features
                    meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                            segment.iloc[:, :feature_idx].values))  # video_id, timestamp, segment_id
                    data[partition]['meta'].append(meta)
                    data[partition]['label'].append(segment.iloc[:, -n_emo_dims:].values)
                    data[partition]['feature'].append(segment.iloc[:, feature_idx:-n_emo_dims].values)

    if normalize:
        idx_list = []

        assert norm_opts is not None and len(norm_opts) == len(feature_set)
        norm_opts = [True if norm_opt == 'y' else False for norm_opt in norm_opts]

        print(f'Feature dims: {feature_dims} ({feature_set})')
        feature_dims = np.cumsum(feature_dims).tolist()
        feature_dims = [0] + feature_dims

        norm_feature_set = []  # normalize data per feature and only if norm_opts is True
        for i, (s_idx, e_idx) in enumerate(zip(feature_dims[0:-1], feature_dims[1:])):
            norm_opt, feature = norm_opts[i], feature_set[i]
            if norm_opt:
                norm_feature_set.append(feature)
                idx_list.append([s_idx, e_idx])

        print(f'Normalized features: {norm_feature_set}')
        data = normalize_data(data, idx_list)

    if save:  # save loaded and preprocessed data
        print('Saving data...')
        pickle.dump(data, open(data_file, 'wb'))

    return data


#if __name__ == '__main__':
#
#    data = load_data(normalize=False)
#    print(pd.DataFrame(data))
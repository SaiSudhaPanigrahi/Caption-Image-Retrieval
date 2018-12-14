import csv
import numpy as np

def parse_features_doc(features_path):
    vec_map = {}
    with open(features_path) as f:
        for row in csv.reader(f):
            img_id = int(row[0].split("/")[1].split(".")[0])
            vec_map[img_id] = np.array([float(x) for x in row[1:]])
    return np.array([v for k, v in sorted(vec_map.items())])

def parse_features(split_idx, num_train, num_dev, num_test, intermediate=False):
	if intermediate == True:
		i_train_dev = parse_features_doc("data/features_train/features_resnet1000intermediate_train.csv")
		i_test = parse_features_doc("data/features_test/features_resnet1000intermediate_test.csv")
	else:
		i_train_dev = parse_features_doc("data/features_train/features_resnet1000_train.csv")
		i_test = parse_features_doc("data/features_test/features_resnet1000_test.csv")
	i_train = i_train_dev[split_idx[:num_train]]
	i_dev = i_train_dev[split_idx[num_train:]]


	print("Built all y matrices!")
	print("i_train shape:", i_train.shape)
	print("i_dev shape:", i_dev.shape)
	print("i_test shape:", i_test.shape)

	return i_train, i_dev, i_test
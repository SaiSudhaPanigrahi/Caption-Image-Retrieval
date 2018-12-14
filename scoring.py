import numpy as np

# takes arg sorted dist
def scoring(dist_idx, num_dev):
    dev_scores = []
    dev_pos_list = []
    for i in range(num_dev):
        dev_pos = np.where(dist_idx[i] == i)
        assert(len(dev_pos) == 1)
        dev_pos_list.append(dev_pos[0])
        if dev_pos[0] < 20:
            # dev_scores.append(1 / (dev_pos[0] + 1))
            dev_scores.append( (21 - dev_pos[0]) / 20 )
        else:
            dev_scores.append(0.0)

    print("Development MAP@20:", np.mean(dev_scores))
    print("Mean index of true image", np.mean(dev_pos_list))
    print("Median index of true image", np.median(dev_pos_list))
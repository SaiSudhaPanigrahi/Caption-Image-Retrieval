import pandas as pd

def write_to_csv(predict, filename, num_predict=20, num_test=2000):
    # convert prediction to '0.jpg'
    predict = predict[:, :num_predict]
    test_predict_str = [None] * num_test
    for i in range(num_test):
        res = ' '.join([str(int(x)) + '.jpg' for x in predict[i]])
        test_predict_str[i] = res # ' '.join([str(int(x)) + '.jpg' for x in test_predict[i]])

    # write to csv
    df = pd.DataFrame(data=test_predict_str)
    df.index = [str(x) + '.txt' for x in range(num_test)]
    df.to_csv(filename, mode='w', index=True, index_label='Descritpion_ID', header=['Top_20_Image_IDs'])
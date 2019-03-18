# Caption-Image-Retrieval

In this project we created an caption-image retrieval model that takes in short sentences as queries and returns the top 20 images that best fit the query. We use a data set that is comprised of images, image descriptions, image tags, and image feature vectors extracted from the last convolutional layers and the last fully connected layers of Deep Residual Network, a state-of-the-art convolutional neural network. Our training data set is made up of 10,000 differen images, each image with five different sentences that describe it. We have 2,000 testing samples. The ReNet feature vectors have 1,000 features for the ones extracted from fully connected layers and 2,049 features for the ones extracted from convolutional layers.

We develop a novel Ridge Regression with L2 Nearest Neighbor Search algorithm that achieves significant accuracy for caption-image retrieval task.

For more information please refer to `caption_image_retrieval.pdf`

## Environment
* Python 2.7
* Numpy
* Scikit-learn
* Matplotlib

---

## Dataset
Please download the dataset from
[word2vec pre-trained Google News corpus word vector model](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)



---

## Running: Best Model
We have tuned the best model parameters and task feature extraction in the `kNN_regression.py` file
Please simply run
```
python ./kNN_regression.py
```
The results will be stored in `./pred_regression.csv`

---

## Other Models
`nearest_neighbor_search.py`

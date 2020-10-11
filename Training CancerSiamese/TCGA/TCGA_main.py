'''
in this script, we will show impact of increasing learning classes and well as
training from scratch and transfer learning
'''
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os, random, pickle
from sklearn.model_selection import train_test_split
import numpy.random as rng
from TCGA_siam import get_siamese_model
from keras.optimizers import Adam
data_path = "/home/UTHSCSA/mostavi/PycharmProjects/FLS_paper_codes/01DataPrep/TCGA_MET_diff_inters_sorted_prim.pkl"

with open(data_path, "rb") as f:
    [TCGA_data, MET_data, difference_TCGA_out, sorted_intersectionTCGA, prim_TCGA]= pickle.load(f)

Number_training_classes = 0   # 0 or 5 or 10
train_classes = set(difference_TCGA_out)
if 0<=Number_training_classes<=10:
    train_classes.update(sorted_intersectionTCGA[:Number_training_classes])

No_train_class = len(set(train_classes))
x_train = TCGA_data.loc[train_classes]
x_train_samples_total, genes_len = TCGA_data.loc[train_classes].shape[0], TCGA_data.loc[train_classes].shape[1]
y_train = x_train.index
y_train_prim = prim_TCGA.index
prim_set = prim_TCGA.index.unique()


x_remain = TCGA_data.loc[sorted_intersectionTCGA[-10:]]
x_test, x_val = train_test_split(x_remain, test_size=0.5)
y_test, y_val = x_test.index, x_val.index

print('Number of training ', x_train_samples_total)


def indices_save(labels):
    class_test_dic_val = {}
    for k in list(set(labels)):
        rng_samples = [i for i, num in enumerate(labels) if num == k]
        class_test_dic_val[k] = rng_samples
    return class_test_dic_val
class_train_ind,class_train_ind_prim, class_test_ind, class_val_ind = list(map(indices_save,[y_train,y_train_prim, y_test, y_val]))

def get_batch(batch_size,md='train'):
    """
    Create batch of n pairs, half same class, half different class
    """

    ## write an exception for batch_size bigger than number of classes
    # randomly sample several classes to use in the batch
    categories = random.sample(set(y_train), No_train_class)  ## scratch

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(No_train_class):
        category = categories[i]
        idx_1 = random.sample(class_train_ind[category],1)[0]

        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1)
        if i >= batch_size // 2:
            category_2 = category
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

        else:
            ind_pop = list(categories).index(category)
            copy_list = categories.copy()
            copy_list.pop(ind_pop)
            category_2 = random.sample(copy_list,1)[0]
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

    return pairs, targets


def make_oneshot_task(N, s="test"):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'val':
        X = x_val.values
        class_test_dic = class_val_ind
    else:
        X = x_test.values
        class_test_dic = class_test_ind

    list_N_samples = random.sample(list(set(y_val)), N)
    true_category = list_N_samples[0]
    out_ind = np.array([random.sample(class_test_dic[j], 2) for j in list_N_samples])
    indices = out_ind[:, 1]
    ex1 = out_ind[0, 0]
    ## create one column of one sample
    test_image = np.asarray([X[ex1]] * N).reshape(N, genes_len, 1)
    support_set = X[indices].reshape(N, genes_len, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set, list_N_samples = shuffle(targets, test_image, support_set, list_N_samples)
    pairs = [test_image, support_set]

    return pairs, targets, true_category, list_N_samples

def test_oneshot(model, N, k, s = "test", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets, true_category, list_N_samples = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct

def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0

def test_nn_accuracy(N_ways, n_trials):
    """Returns accuracy of NN approach """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials, N_ways))

    n_right = 0
    for i in range(n_trials):
        pairs, targets, true_category, list_N_samples = make_oneshot_task(N_ways, "test")
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct
    return 100.0 * n_right / n_trials


input_shape = (genes_len, 1)
model = get_siamese_model(input_shape)
model.summary()
optimizer = Adam(lr = 0.000005)

Trans = 'Transfer_learning' 

model.compile(loss="binary_crossentropy",optimizer=optimizer)
model_path = "/home/UTHSCSA/mostavi/PycharmProjects/FSL_revise_third/01main_train_test/saved_model_weights/"

model_json = model.to_json()
with open(model_path + "model_siam.json", "w") as json_file:
    json_file.write(model_json)


# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 128   # max 12 for 19
n_iter = 20000 # No. of training iterations
N_way = 10 # how many classes for testing one-shot tasks
n_val = 1000 # how many one-shot tasks to validate on
best = -1


model_name ='TCGA_trans_'+'{}'.format(len(train_classes))
print("Starting training process!")
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Train Loss: {0}".format(loss))
        val_acc = test_oneshot(model, N_way, n_val, s='val', verbose=True)
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            model.save_weights(os.path.join(model_path, 'weights_' + model_name + '.h5'))
            print(str(i))
            best = val_acc




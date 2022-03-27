# Implement a covariate shift detector and corrector using
# the FashionMNIST dataset

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


# ---------- Set Parameters and prepare data according to desired distributions ---------- #

# Binary classifier parameters
batch_size_bc = 256
num_hidden_bc = 256
num_outputs_bc = 2
lr_bc = 0.25
num_epochs_bc = 25

# Regular and covariate-shift-corrected classifier parameters
batch_size = 256
num_hidden = 64
num_outputs = 10
lr = 0.25
num_epochs = 50

# Define training and test dataset distributions (fraction of each category to keep)
train_prob = np.ones(10)
test_prob = np.ones(10)


def prepare_initial_data(train_prob, test_prob):
    """Generate training and testing fashionMNIST datasets according to
     specified probability distributions over categories.
     Inputs:
        train_prob: array specifying fraction of each category to retain
            in training set.
        test_prob: array specifying fraction of each category to retain
            in test set.
    Outputs:
        train_new: new training set of specified distribution
            over categories.
        test_new: new test set of specified distribution over
            categories."""
    # ---- Load and process dataset ---- #

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    # ---- Construct new train dataset ---- #

    # Sort the data
    sorted_id = tf.argsort(y_train)
    x_train, y_train = tf.gather(x_train, sorted_id), tf.gather(y_train, sorted_id)

    # Create new training data with specified distribution
    num_keep = np.round(6000 * train_prob).astype(int)
    cum_num_keep = np.concatenate((np.array([0]), np.cumsum(num_keep)))
    keep_id = np.zeros(np.sum(num_keep), dtype=int)
    for i in range(len(num_keep)):
        keep_id[cum_num_keep[i]:cum_num_keep[i+1]] = np.arange(i*6000, i*6000 + num_keep[i])
    x_train, y_train = tf.gather(x_train, keep_id), tf.gather(y_train, keep_id)

    # ---- Construct new test dataset ---- #

    # Sort the data
    sorted_id = tf.argsort(y_test)
    x_test, y_test = tf.gather(x_test, sorted_id), tf.gather(y_test, sorted_id)

    # Create new test data with specified distribution
    num_keep = np.round(1000 * test_prob).astype(int)
    cum_num_keep = np.concatenate((np.array([0]), np.cumsum(num_keep)))
    keep_id = np.zeros(np.sum(num_keep), dtype=int)
    for i in range(len(num_keep)):
        keep_id[cum_num_keep[i]:cum_num_keep[i+1]] = np.arange(i * 1000, i * 1000 + num_keep[i])
    x_test, y_test = tf.gather(x_test, keep_id), tf.gather(y_test, keep_id)

    return x_train, y_train, x_test, y_test


def shuffle_data(x, y):
    """Shuffle (randomize) examples in dataset (y, x)."""
    # Shuffle both datasets
    rand_id = tf.random.shuffle(np.arange(tf.shape(x)[0]))
    X, Y = tf.gather(x, rand_id), tf.gather(y, rand_id)
    return X, Y


def prepare_bc_data(x_train, y_train, x_test, y_test, batch_size, tt_split=0.8):
    """Create (features, distribution_class)-dataset for the training(=0)
    and testing distributions=(1) in randomized batches of size 'batch_size'
    with a 'tt_split' training/testing split.
    Inputs:
        x_train, y_train: features and labels from training set.
        x_test, y_test: features and labels from test set.
        batch_size: batch size for data iterator.
        tt_split: fraction of all samples which should be in the new training set.
    Output:
        bc_train_iter: training dataset iterator returning batches of 'batch_size'.
        bs_test_iter: test ""                                                   ""."""

    # Shuffle both datasets
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test = shuffle_data(x_test, y_test)
    y_train, y_test = tf.zeros_like(y_train), tf.ones_like(y_test)

    # Reduce dataset to so there are equal samples from each distribution
    num_samples = np.min((tf.shape(x_train)[0], tf.shape(x_test)[0]))
    y_train, y_test = y_train[0:num_samples], y_test[0:num_samples]
    X_train, X_test = x_train[0:num_samples,], x_test[0:num_samples,]
    y = tf.concat([y_train, y_test], 0)
    X = tf.concat([X_train, X_test], 0)

    # Create training/testing split
    num_train = round(len(X) * tt_split)

    # Shuffle all samples and split into training/testing sets
    rand_id = tf.random.shuffle(np.arange(tf.shape(X)[0]))
    train_id = rand_id[0:num_train]
    test_id = rand_id[num_train:]
    X_train, y_train = tf.gather(X, train_id), tf.gather(y, train_id)
    X_test, y_test = tf.gather(X, test_id), tf.gather(y, test_id)

    # Create data iterators
    bc_train_iter = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    bc_test_iter = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    return bc_train_iter, bc_test_iter


def prepare_mlp_data(x_train, y_train, x_test, y_test, batch_size):
    """Create (features, labels)-dataset for the training and testing sets
     in randomized batches of size 'batch_size' with a 'tt_split'
     training/testing split.
    Inputs:
        x_train, y_train: features and labels from training set.
        x_test, y_test: features and labels from test set.
        batch_size: batch size for data iterator.
        tt_split: fraction of all samples which should be in the new training set.
    Output:
        train_iter: training dataset iterator returning batches of 'batch_size'.
        test_iter: test ""                                                   ""."""

    # Shuffle both datasets
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test = shuffle_data(x_test, y_test)

    # Create data iterators
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_iter, test_iter


def get_cc_weights(bc_net, X, c):
    """Obtain covariate shift correction weights."""
    if bc_net == None:
        cc_weights = tf.ones(X.shape[0])
    else:
        cc_weights = bc_net(X)
        cc_weights = tf.math.minimum(tf.math.exp((cc_weights[:, 1] - cc_weights[:, 0])), c)
    return cc_weights


def evaluate_loss(net, data_iter, loss):
    """Evaluate the accuracy of 'net' on data_iter using 'loss'."""
    num_examples = 0
    l_sum = 0
    for X, y in data_iter:
        l = loss(y, net(X))
        l_sum += l
        num_examples += y.shape[0]
    return l_sum / num_examples


def evaluate_accuracy(net, data_iter):
    """Count number of correct predictions."""
    num_examples = 0
    num_correct = 0
    for X, y in data_iter:
        y_hat = tf.math.argmax(net(X), axis=1)
        cmp = tf.cast(y_hat, y.dtype) == y
        num_correct += tf.reduce_sum(tf.cast(cmp, 'int32'))
        num_examples += y.shape[0]
    return num_correct / num_examples


def train_net(net, train_iter, test_iter, trainer, loss, num_epochs=25, bc_net=None, c=10.0):
    """Train a network for classification, with optional covariate shift correction."""
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        print(epoch)
        for X, y in train_iter:
            cc_weights = get_cc_weights(bc_net, X, c)
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat, sample_weight=cc_weights)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        train_loss.append(evaluate_loss(net, train_iter, loss))
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return train_loss, train_acc, test_acc


def get_net(num_output, num_hidden=256):
    """Create a 1-layer mlp network."""
    net = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_hidden, activation='relu'),
        tf.keras.layers.Dense(num_output)
    ])
    return net


# ---------- Prepare dataset with desired distribution ---------- #

# Generate initial training and test datasets
x_train, y_train, x_test, y_test = prepare_initial_data(train_prob, test_prob)


# ---------- Train/Test binary classifier to distinguish between train and test sets ---------- #

# Generate dataset iterator for binary classification task
bc_train_iter, bc_test_iter = prepare_bc_data(x_train, y_train, x_test, y_test, batch_size_bc)

# Create binary classifier
bc_net = get_net(num_outputs_bc)
bc_trainer = tf.keras.optimizers.SGD(lr_bc)
bc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Train/Test binary classifier
bc_train_loss, bc_train_acc, bc_test_acc = train_net(bc_net, bc_train_iter, bc_test_iter, bc_trainer,
                                  bc_loss, num_epochs_bc)


# ---------- Train/Test regular and covariate-shift-corrected MLPs ----------- #

# Create train/test data iterators
train_iter, test_iter = prepare_mlp_data(x_train, y_train, x_test, y_test, batch_size)

# Create MLP for FashionMNIST classification
net_reg = get_net(num_outputs, num_hidden)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(lr)

# Train/Test MLP without covariate shift correction
train_loss, train_acc, test_acc = train_net(net_reg, train_iter, test_iter, trainer, loss, num_epochs)

# Train/Test MLP with covariate shift correction
net_cc = get_net(num_outputs, num_hidden)
train_loss_cc, train_acc_cc, test_acc_cc = train_net(net_cc, train_iter, test_iter, trainer, loss, num_epochs, bc_net)


# ---------- Plot results --------- #

fig, ax = plt.subplots()
ax.plot(np.arange(num_epochs), train_acc, '-sr', label='Training Accuracy Regular NN')
ax.plot(np.arange(num_epochs), train_acc_cc, '-or', label='Training Accuracy Covariate-Shift-Corrected NN')
ax.plot(np.arange(num_epochs), test_acc, '-sb', label='Testing Accuracy Regular NN')
ax.plot(np.arange(num_epochs), test_acc_cc, '-ob', label='Testing Accuracy Covariate-Shift-Corrected NN')
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Prediction Accuracy')
ax.grid()
ax.legend()

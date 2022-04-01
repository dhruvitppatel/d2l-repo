import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# ---------- Set Parameters + Build Models ---------- #

loss = tf.keras.losses.MeanSquaredError()

# Linear regression model
k_lin_reg = 5                       # cross-validation folds
num_epochs_lin_reg = 100
lr_lin_reg = 5                      # learning rate
lam_lin_reg = 0                     # L2 regularization parameter
batch_size_lin_reg = 64

optimizer_lr = tf.keras.optimizers.Adam(lr_lin_reg)

# NN Model
k_nn = 5                            # cross-validation folds
num_epochs_nn = 200
lr_nn = 0.05                        # learning rate
lam_nn = 1e-6                       # L2 regularization parameter
batch_size_nn = 64
num_hidden = 16                     # num. hidden units

optimizer_nn = tf.keras.optimizers.Adam(lr_nn)


def get_lin_reg_model():
    net = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(lam_lin_reg))
    ])
    return net


def get_nn_net():
    net = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(lam_nn)),
        tf.keras.layers.Dense(1)
    ])
    return net

# ---------- Load + Pre-process Data ---------- #

train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

train_features, train_labels = train_data.iloc[:, 1:-1], train_data.iloc[:, -1]
test_features = test_data.iloc[:, 1:]

# Standardize values of numeric training features
numeric_features_id = train_features.dtypes[train_features.dtypes != 'object'].index
train_features[numeric_features_id] = train_features[numeric_features_id].apply(
    lambda x: (x - x.mean()) / x.std())
test_features[numeric_features_id] = train_features[numeric_features_id].apply(
    lambda x: (x - x.mean()) / x.std())

# Fill missing values with mean (=0)
train_features[numeric_features_id] = train_features[numeric_features_id].fillna(0)
test_features[numeric_features_id] = test_features[numeric_features_id].fillna(0)

# Convert categorical features to one-hot encoding
n_train = train_features.shape[0]
all_features = pd.get_dummies(pd.concat((train_features, test_features)), dummy_na=True)
train_features = all_features.iloc[:n_train, :]
test_features = all_features.iloc[n_train:, :]

# Convert data to tensors
train_features = tf.constant(train_features, dtype=tf.float32)
train_labels = tf.constant(train_labels, dtype=tf.float32)
test_features = tf.constant(test_features, dtype=tf.float32)

# ---------- Prepare Training/Testing Routines ---------- #


def load_data_iterators(x, y, batch_size, is_train=True):
    """Create tensforflow data iterators."""
    data_iter = tf.data.Dataset.from_tensor_slices((x, y))
    if is_train:
        data_iter = data_iter.shuffle(buffer_size=1000)
    data_iter = data_iter.batch(batch_size)
    return data_iter


def log_rmse(y, y_hat):
    y_hat_clip = tf.clip_by_value(y_hat, 1, float('inf'))
    l = tf.math.sqrt(tf.reduce_mean(loss(tf.math.log(y), tf.math.log(y_hat_clip))))
    return l


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, optimizer,
          batch_size):
    train_ls, test_ls = [], []
    train_iter = load_data_iterators(train_features, train_labels, batch_size)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X, training=True) + net.losses
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls


def k_fold(k, train_features, train_labels, num_epochs, optimizer,
           batch_size, model='linear', split=0.2, seed=100):
    train_ls_sum, valid_ls_sum = 0, 0
    for i in range(k):
        x_train, x_test, y_train, y_test = train_test_split(train_features.numpy(), train_labels.numpy(),
                                                            test_size=split, random_state=seed+i)
        net = get_lin_reg_model() if model == 'linear' else get_nn_net()
        train_ls, valid_ls = train(net, x_train, y_train, x_test, y_test, num_epochs, optimizer,
                                   batch_size)
        train_ls_sum += train_ls[-1]
        valid_ls_sum += valid_ls[-1]
        if i == 0:
            fig, ax = plt.subplots()
            ax.plot(list(range(1, num_epochs+1)), train_ls, '-*k', label='train')
            ax.plot(list(range(1, num_epochs+1)), valid_ls, '-*r', label='valid')
            ax.set_xlabel('epoch')
            ax.set_ylabel('rmse')
            ax.set_yscale('log')
            ax.legend()
            ax.grid()
            plt.title('Training and Validation log rmse on training set')
        print(f'fold {i+1}, train log rmse {float(train_ls[-1]):f}',
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_ls_sum / k, valid_ls_sum / k


def train_predict(train_features, train_labels, test_features, num_epochs, optimizer,
                  batch_size, model='linear'):
    net = get_lin_reg_model() if model == 'linear' else get_nn_net()
    train_ls, _ = train(net, train_features, train_labels, test_features, None, num_epochs,
                        optimizer, batch_size)
    fig, ax = plt.subplots()
    ax.plot(list(range(1, num_epochs+1)), train_ls, '-*k')
    ax.set_xlabel('epoch')
    ax.set_ylabel('log rmse')
    ax.grid()
    plt.title('Training log rmse on test set')
    preds = net(test_features).numpy()
    # Reformat predictions to export to kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


# ---------- Train and Cross-Validate ---------- #

# Linear regression model
train_lin_reg, valid_lin_reg = k_fold(k_lin_reg, train_features, train_labels, num_epochs_lin_reg,
                          optimizer_lr, batch_size_lin_reg, model='linear')

# NN model
train_nn, valid_nn = k_fold(k_nn, train_features, train_labels, num_epochs_nn,
                            optimizer_nn, batch_size_nn, model='nn')


# --------- Train and Predict --------- #

# # Linear model
# train_predict(train_features, train_labels, test_features, num_epochs_lin_reg, optimizer_lr,
#               batch_size_lin_reg, model='linear')

# # NN model
# train_predict(train_features, train_labels, test_features, num_epochs_nn, optimizer_nn,
#               batch_size_nn, model='nn')

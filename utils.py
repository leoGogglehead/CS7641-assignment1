import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import (
    cross_validate, 
    validation_curve
)
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
# import timeit

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12


# set stype
plt.style.use("seaborn-darkgrid")
my_palette = [
    '#9F2B68', '#f64a8a', '#fdab9f', '#F8DE7E', '#7dcfb6', '#00b2ca', '#1d4e89', '#7F00FF'
    ] 

# get environment var
DATASET = os.getenv("dataset")


def plot_validation_curve(
    clf,
    X: np.array,
    y: np.array,
    param_name: str,
    param_range: np.array,
    metric: str = 'neg_log_loss',
    title: str = "Insert Title"
):
    """
    Function to plot validation curve.

    Args:
        clf (_type_): sklearn classifier
        X (_type_): training matrix array
        y (_type_): label array
        param_name (_type_): 
        metric (str, optional): _description_. Defaults to 'neg_log_loss'.
        title (str, optional): _description_. Defaults to "Insert Title".
    """

    # get training and cross validation scores before taking the mean
    train_scores, val_scores = validation_curve(
        estimator=clf,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring=metric)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    if type(param_range[0]) == tuple:
        param_range = [i[0] for i in param_range]

    # plotting 
    plt.figure(figsize=(8, 6))
    plt.title("Validation Curve: " + title, fontsize=17)
    plt.xlabel(f"{param_name} Range", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.fill_between(
        param_range,
        train_mean - 2*train_std,
        train_mean + 2*train_std,
        alpha=0.1,
        color=my_palette[0]
        )
    plt.fill_between(
        param_range,
        val_mean - 2*val_std,
        val_mean + 2*val_std,
        alpha=0.1,
        color=my_palette[-1]
        )
    plt.plot(
        param_range,
        train_mean,
        'o-',
        color=my_palette[0],
        label="Training"
        )
    plt.plot(
        param_range,
        val_mean,
        '^-',
        color=my_palette[-1],
        label="CV"
        )
    plt.xticks(fontsize=10.5, rotation=30)
    plt.yticks(fontsize=10.5)
    plt.legend(loc="best", fontsize=11)
    plt.savefig(f"figs/{DATASET}/val_curve_{title}.png", bbox_inches="tight")
    plt.show()


def plot_learning_curve(
    clf,
    X: np.array,
    y: np.array,
    train_sizes: np.array = None,
    metric: str = 'neg_log_loss',
    title: str = "Insert Title"
):
    """
    Function to plot learning curves and return train and validation scores

    Args:
        clf (_type_): _description_
        X (np.array): _description_
        y (np.array): _description_
        train_sizes (np.array): _description_
        metric (str, optional): _description_. Defaults to 'neg_log_loss'.
        title (str, optional): _description_. Defaults to "Insert Title".
    """

    # traing and val scores
    train_mean, train_std = [], []
    val_mean, val_std = [], []

    # time for fitting and prediction
    fittime_mean, fittime_std = [], []
    predtime_mean, predtime_std = [], [] 

    if train_sizes is None:
        train_sizes = (np.linspace(.05, 1.0, 20) * y.shape[0]).astype('int')
    else:
        train_sizes = (train_sizes * y.shape[0]).astype('int')

    # loop over all training sizes, for each size get get the train
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_sub, y_sub = X[idx, :], y[idx]
        scores = cross_validate(
            clf,
            X_sub,
            y_sub,
            cv=5,
            scoring=metric,
            return_train_score=True
            )

        # metric scoring
        train_mean.append(np.mean(scores['train_score']))
        train_std.append(np.std(scores['train_score']))
        val_mean.append(np.mean(scores['test_score']))
        val_std.append(np.std(scores['test_score']))

        # wall time
        fittime_mean.append(np.mean(scores['fit_time']))
        fittime_std.append(np.std(scores['fit_time']))
        predtime_mean.append(np.mean(scores['score_time']))
        predtime_std.append(np.std(scores['score_time']))

    _plot_LC(
        train_sizes,
        np.array(train_mean),
        np.array(train_std),
        np.array(val_mean),
        np.array(val_std),
        metric,
        title)

    _plot_times(
        train_sizes,
        np.array(fittime_mean),
        np.array(fittime_std),
        np.array(predtime_mean),
        np.array(predtime_std),
        title)


def _plot_LC(train_sizes, train_mean, train_std, val_mean, val_std, metric, title):
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve: " + title, fontsize=17)
    plt.xlabel("Sample Size", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.fill_between(
        train_sizes,
        train_mean - 2*train_std,
        train_mean + 2*train_std,
        alpha=0.1,
        color=my_palette[0]
        )
    plt.fill_between(
        train_sizes,
        val_mean - 2*val_std,
        val_mean + 2*val_std,
        alpha=0.1,
        color=my_palette[-1]
        )
    plt.plot(train_sizes, train_mean, 'o-', color=my_palette[0], label="Training")
    plt.plot(train_sizes, val_mean, '^-', color=my_palette[-1], label="CV")
    plt.xticks(fontsize=10.5, rotation=30)
    plt.yticks(fontsize=10.5)
    plt.legend(loc="best", fontsize=11)
    plt.savefig(f"figs/{DATASET}/learn_curve_{title}.png", bbox_inches="tight")
    plt.show()
    

def _plot_times(train_sizes, train_mean, train_std, pred_mean, pred_std, title):
    plt.figure(figsize=(8, 6))
    plt.title("Time: " + title, fontsize=17)
    plt.xlabel("Sample Size", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.fill_between(
        train_sizes,
        train_mean - 2*train_std,
        train_mean + 2*train_std,
        alpha=0.1,
        color=my_palette[0]
        )
    plt.fill_between(
        train_sizes,
        pred_mean - 2*pred_std,
        pred_mean + 2*pred_std,
        alpha=0.1,
        color=my_palette[-1]
        )
    plt.plot(train_sizes, train_mean, 'o-', color=my_palette[0], label="Training")
    plt.plot(train_sizes, pred_mean, '^-', color=my_palette[-1], label="CV infer")
    plt.xticks(fontsize=10.5, rotation=30)
    plt.yticks(fontsize=10.5)
    plt.legend(loc="best", fontsize=11)
    plt.savefig(f"figs/{DATASET}/wall_time_{title}.png", bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
        conf_mat,
        classes=None,
        normalize=False,
        title="Insert Title"):

    if not classes:
        classes = conf_mat.shape[0]

    if normalize:
        conf_mat = conf_mat/conf_mat.sum(axis=1).reshape(-1, 1)

    cmap = sns.light_palette("steelblue", as_cmap=True)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_mat, annot=True, cmap=cmap)
    ax.set_xlabel("Predicted Labels")
    ax.xaxis.set_ticklabels(classes)
    ax.set_ylabel("True Labels")
    ax.yaxis.set_ticklabels(classes)
    ax.set_title('Confusion Matrix: ' + title, fontsize=17)
    plt.savefig(f"figs/{DATASET}/conf_mat_{title}.png", bbox_inches="tight")
    plt.show()
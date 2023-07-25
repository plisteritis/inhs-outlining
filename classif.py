from inhs_outlining import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_mat(mat_file):
    data = np.genfromtxt(mat_file, dtype=str, delimiter=',')
    np.random.seed(0)
    np.random.shuffle(data)
    # Data matrix CSVs contain a locus X&Y in the first two columns, but we ignore those in this function
    #regular:
    return data[:, 2:-1].astype(float), data[:, -1]

def load_mat_with_id(mat_file):
    #should work with _new.csv files for now...
    data = np.genfromtxt(mat_file, dtype=str, delimiter=',')
    np.random.seed(0)
    np.random.shuffle(data)
    # Data matrix CSVs contain a locus X&Y in the first two columns, but we ignore those in this function
    #for id's:
    return data[:, 2:-2].astype(float), data[:, -2], data[:, -1]


def normalize(X):
    Xn = []
    for efds in X:
        xni, trans = pyefd.normalize_efd(efds.reshape(-1, 4), return_transformation=True)
        Xn.append(xni.flatten()[3:] * (-1 if abs(trans[1]) > 0.25 else 1))
    return np.array(Xn)


def make_top_k_scorer(k):
    return lambda clf, X, Y: top_k_accuracy_score(Y, clf.predict_proba(X), k=k)


f1_scorer = lambda clf, X, Y: f1_score(Y, clf.predict(X), average="weighted")
precision_scorer = lambda clf, X, Y: precision_score(Y, clf.predict(X), average="weighted")
recall_scorer = lambda clf, X, Y: recall_score(Y, clf.predict(X), average="weighted")
pipeline = lambda clf: Pipeline([
    ("standardize", StandardScaler()),
    ("reduce", LDA()),
    ("classify", clf)
])


def synthesize_n_rows_from(real_rows, n):
    synth_rows = np.empty((n, real_rows.shape[1]))
    for i in range(n):
        num_refs = np.random.randint(2, real_rows.shape[0] + 1)
        np.random.shuffle(real_rows)
        refs = real_rows[:num_refs, :]
        weights = np.random.dirichlet(np.ones(num_refs), size=1)
        synth_rows[i] = np.sum(refs * weights.T, axis=0)
    return np.array(synth_rows)


def ensure_at_least_n_rows_per_label(X, Y, n):
    Xsynth = np.empty((0, X.shape[1]))
    Ysynth = np.empty(0)
    for label, count in zip(*np.unique(Y, return_counts=True)):
        rows_needed = max(0, n - count)
        Xsynth = np.concatenate((Xsynth, synthesize_n_rows_from(X[Y == label], rows_needed)), axis=0)
        Ysynth = np.concatenate((Ysynth, np.repeat(label, rows_needed)))
    return np.concatenate((X, Xsynth), axis=0), np.concatenate((Y, Ysynth))


def run_cv_trials(clf, X, Y, folds=5, score=make_top_k_scorer(1), min_rows_per_label=0):
    print("Model:", clf)
    np.random.seed(0)
    scores = []
    kf = KFold(n_splits=folds)
    for (train_inds, test_inds) in kf.split(X):
        Xtrain, Ytrain = X[train_inds], Y[train_inds]
        Xtrain, Ytrain = ensure_at_least_n_rows_per_label(Xtrain, Ytrain, min_rows_per_label)
        Xmean = np.mean(Xtrain, axis=0)
        Xstd = np.std(Xtrain, axis=0)
        Xstd[Xstd == 0] = 1
        Xtrain = (Xtrain - Xmean) / Xstd
        lda = LDA()
        Xtrain = lda.fit_transform(Xtrain, Ytrain)
        clf.fit(Xtrain, Ytrain)
        Xtest = ((X[test_inds] - Xmean) / Xstd) @ lda.scalings_
        scores.append(score(clf, Xtest, Y[test_inds]))
    scores = np.array(scores)
    print("avg:   %.1f%%" % (scores.mean() * 100))
    print("std:   %.1f%%" % (scores.std() * 100))


def show_confusion_mat(clf, X, Y, folds=5):
    # Note that the predictions shown might not match what the classifier would predict in a cross_val_score() trial.
    y_pred = cross_val_predict(pipeline(clf), X, Y, cv=folds)
    classes = np.unique(Y)
    cm = confusion_matrix(Y, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Greys", xticks_rotation=45)
    showplt()

#ajani testing stuff: trying to see what just confusion matrix can do...
def show_notro_phena_fails(clf, X, Y, folds=5):
    y_pred = cross_val_predict(pipeline(clf), X, Y, cv=folds)
    classes = np.unique(Y)
    cm = confusion_matrix(Y, y_pred, labels=classes)
    #failures gives number of times true label was done wrong by society, i mean, the model
    #  if i return indices where y_pred was wrong...ima be... 
    failures = np.column_stack(np.unique(Y[y_pred != Y], return_counts = True))
    indices = [[], []]
    for index in range(len(Y)):
        if Y[index] == "Notropis" and y_pred[index] != "Notropis":
            indices[0].append(index)
        elif Y[index] == "Phenacobius" and y_pred[index] != "Phenacobius":
            indices[1].append(index)
    notropis_index = (indices[0], len(indices[0]))
    phenacobius_index = (indices[1], len(indices[1]))
    print(notropis_index, phenacobius_index)


def show_variation(fishes):
    n = max(len(fish.normalized_outline) for fish in fishes)
    all_efds = np.array(pad_ragged([list(fish.encoding[0].ravel()) for fish in fishes]))
    avg = np.mean(all_efds, axis=0)
    std = np.sqrt(((all_efds - avg) ** 2).sum(axis=0) / len(fishes))
    recons = lambda efds: reconstruct(efds.reshape(-1, 4), n, (0, 0))
    contours = [recons(efds) for efds in [avg, avg + std, avg - std]]
    show_contour(*contours, colors=[(0xff, 0, 0), (0, 0, 0xff), (0xff, 0xff, 0xff)])

def get_ids(mat_file):
    return load_mat_with_id(mat_file)[2]

if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    #testX, testY, testID = load_mat("1mm_seven_genera_new.csv")
    X, Y = load_mat("1mm_seven_genera.csv")
    Xn = normalize(X)

    svm = SVC(random_state=0, kernel='linear', C=0.1, probability=True)
    #knn = KNeighborsClassifier(n_neighbors=7)
    #run_cv_trials(knn, Xn, Y, folds=5) 
    flops = show_notro_phena_fails(svm, Xn, Y)
    

    x2, y2, ids = load_mat_with_id("1mm_seven_genera_new.csv")
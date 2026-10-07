"""Step 4: Signal classification (SVM, LDA, DBN-style RBM pipeline)."""
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=50)


def run_svm(X, y):
    clf = svm.SVC(kernel="rbf", gamma="auto", C=0.00004)
    scores = cross_val_score(clf, X, y, cv=5)
    print("SVM CV scores:", scores, "mean:", scores.mean())
    X_tr, X_te, y_tr, y_te = split(X, y)
    clf.fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))
    return clf


def run_lda(X, y):
    X_tr, X_te, y_tr, y_te = split(X, y)
    lda = LinearDiscriminantAnalysis().fit(X_tr, y_tr)
    y_pred = lda.predict(X_te)
    print(classification_report(y_te, y_pred, zero_division=0))
    print("LDA accuracy:", accuracy_score(y_te, y_pred))
    scores = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5)
    print("LDA CV scores:", scores, "mean:", scores.mean())


def run_dbn(X, y, n_layers=3, rbm_components=100):
    rbms = [(f"rbm_layer{i}", BernoulliRBM(n_components=rbm_components, n_iter=10, random_state=42))
            for i in range(n_layers)]
    dbn = Pipeline([("scaler", StandardScaler()), *rbms,
                    ("logistic", LogisticRegression(solver="lbfgs", max_iter=10000,
                                                    C=0.000004, random_state=42))])
    X_tr, X_te, y_tr, y_te = split(X, y)
    dbn.fit(X_tr, y_tr)
    print("DBN accuracy:", dbn.score(X_te, y_te))

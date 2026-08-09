
from flask import Flask, render_template, request, redirect, url_for, flash
import os, uuid, re, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score,
    cohen_kappa_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "heart-disease-data-mining"
UPLOAD_DIR = os.path.join("static", "generated")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED = {"arff", "csv"}

def read_arff(path):
    rows, attrs, in_data = [], [], False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                m = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)", line, re.I)
                if m:
                    name = next(x for x in m.groups()[:3] if x is not None)
                    typ = m.group(4).strip()
                    attrs.append((name, typ))
            elif low.startswith("@data"):
                in_data = True
            elif in_data and not line.startswith("@"):
                # Basic CSV/ARFF row parser; handles quoted commas.
                parts = re.split(r',(?=(?:[^"\']|"[^"]*"|\'[^\']*\')*$)', line)
                rows.append([p.strip().strip('"').strip("'") for p in parts])
    cols = [a[0] for a in attrs]
    df = pd.DataFrame(rows, columns=cols if cols else None)
    return df

def load_dataset(path):
    if path.lower().endswith(".arff"):
        df = read_arff(path)
    else:
        df = pd.read_csv(path)
    df = df.dropna(how="all").reset_index(drop=True)
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least two columns.")
    # Use the last column as target, matching the usual UCI/Weka ARFF layout.
    target = df.columns[-1]
    return df, target

def prepare_xy(df, target):
    X = df.drop(columns=[target]).copy()
    y_raw = df[target].astype(str).str.strip()
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return X, y, enc

def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", __import__("sklearn").preprocessing.OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return ColumnTransformer(transformers=transformers)

def models():
    return {
        "J48 Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "KNN": Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("knn", KNeighborsClassifier(n_neighbors=3))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

def fit_model(name, X, y):
    pre = build_preprocessor(X)
    model = models()[name]
    # GaussianNB cannot consume sparse matrices, so densify via a special path.
    if name == "Naive Bayes":
        from sklearn.preprocessing import OneHotEncoder
        numeric = X.select_dtypes(include=np.number).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers=[]
        if numeric:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical))
        pre = ColumnTransformer(transformers=transformers)
    pipe = Pipeline([("prep", pre), ("model", model)])
    return pipe

def make_outputs(name, df, target, test_size=0.30):
    X, y, enc = prepare_xy(df, target)
    if len(np.unique(y)) < 2:
        raise ValueError("The target column must contain at least two classes.")
    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat
    )
    pipe = fit_model(name, X, y)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

    labels = list(range(len(enc.classes_)))
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(
        y_test, pred, labels=labels, target_names=[str(x) for x in enc.classes_],
        zero_division=0, output_dict=True
    )
    acc = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)

    uid = uuid.uuid4().hex
    cm_path = os.path.join(UPLOAD_DIR, f"{uid}_cm.png")
    roc_path = os.path.join(UPLOAD_DIR, f"{uid}_roc.png")
    tree_path = os.path.join(UPLOAD_DIR, f"{uid}_tree.png")

    # Confusion matrix
    fig = plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(enc.classes_))
    plt.xticks(ticks, enc.classes_, rotation=45)
    plt.yticks(ticks, enc.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    # ROC curve (one-vs-rest for multiclass)
    fig = plt.figure(figsize=(7, 5))
    if proba is not None:
        if len(enc.classes_) == 2:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1], pos_label=1)
            plt.plot(fpr, tpr, label=f"AUC = {auc(fpr,tpr):.3f}")
        else:
            y_bin = label_binarize(y_test, classes=labels)
            for i, cls in enumerate(enc.classes_):
                if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                plt.plot(fpr, tpr, label=f"Class {cls} AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],"--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)

    # Tree visualization for J48-equivalent tree
    tree_url = None
    if name == "J48 Decision Tree":
        prep = pipe.named_steps["prep"]
        Xt = prep.transform(X_train)
        feature_names = list(prep.get_feature_names_out())
        estimator = pipe.named_steps["model"]
        fig = plt.figure(figsize=(18, 10))
        plot_tree(estimator, feature_names=feature_names, class_names=[str(x) for x in enc.classes_],
                  filled=True, max_depth=4, fontsize=7)
        plt.title("J48 Decision Tree Visualization (sklearn entropy tree)")
        plt.tight_layout()
        plt.savefig(tree_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        tree_url = "/" + tree_path.replace("\\", "/")

    metrics = {
        "accuracy": acc*100,
        "kappa": kappa,
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "correct": int((y_test == pred).sum()),
        "incorrect": int((y_test != pred).sum()),
        "total_test": int(len(y_test)),
        "total_dataset": int(len(df))
    }
    complete = (
        f"Algorithm: {name}\n"
        f"Dataset instances: {len(df)}\n"
        f"Training instances: {len(y_train)}\n"
        f"Testing instances: {len(y_test)}\n"
        f"Accuracy: {acc*100:.2f}%\n"
        f"Kappa Statistic: {kappa:.4f}\n"
        f"Weighted Precision: {metrics['precision']:.4f}\n"
        f"Weighted Recall: {metrics['recall']:.4f}\n"
        f"Weighted F-Measure: {metrics['f1']:.4f}\n\n"
        + classification_report(y_test, pred, labels=labels,
                                target_names=[str(x) for x in enc.classes_],
                                zero_division=0)
        + "\nConfusion Matrix:\n" + np.array2string(cm)
    )
    return metrics, report, enc.classes_.tolist(), cm_path, roc_path, tree_url, complete

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please choose an ARFF or CSV dataset.")
            return redirect(url_for("home"))
        ext = file.filename.rsplit(".",1)[-1].lower()
        if ext not in ALLOWED:
            flash("Please upload an .arff or .csv file.")
            return redirect(url_for("home"))
        uid = uuid.uuid4().hex
        path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
        file.save(path)
        try:
            df, target = load_dataset(path)
            return render_template("choose.html", filename=file.filename, rows=len(df),
                                   columns=list(df.columns), target=target, token=uid)
        except Exception as e:
            flash(f"Could not read dataset: {e}")
            return redirect(url_for("home"))
    return render_template("home.html")

@app.route("/run", methods=["POST"])
def run():
    token = request.form["token"]
    files = [x for x in os.listdir(UPLOAD_DIR) if x.startswith(token+"_")]
    if not files:
        flash("Uploaded dataset not found. Please upload again.")
        return redirect(url_for("home"))
    path = os.path.join(UPLOAD_DIR, files[0])
    name = request.form.get("algorithm", "J48 Decision Tree")
    try:
        df, target = load_dataset(path)
        metrics, report, classes, cm_path, roc_path, tree_url, complete = make_outputs(name, df, target)
        return render_template(
            "results.html", filename=files[0].split("_",1)[1], algorithm=name,
            metrics=metrics, report=report, classes=classes,
            cm_url="/"+cm_path.replace("\\","/"),
            roc_url="/"+roc_path.replace("\\","/"),
            tree_url=tree_url, complete=complete,
            columns=list(df.columns), preview=df.head(10).to_html(classes="data-table", index=False)
        )
    except Exception as e:
        flash(f"Could not run algorithm: {e}")
        return redirect(url_for("home"))

@app.route("/reset")
def reset():
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)

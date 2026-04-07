"""
Analyse pedagogique des donnees Data_Cannabis.csv.

Objectifs:
1) Importation des donnees
2) Visualisation exploratoire
3) PCA et visualisation
4) Classification supervisee

Le script est volontairement simple et modifiable par des etudiants.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Parametres faciles a modifier
RANDOM_STATE = 42
TEST_SIZE = 0.25
DATA_PATH = Path("labs/data/Data_Cannabis.csv")
OUTPUT_DIR = Path("labs/figs")
TARGET_COLUMN = "Etat"


def load_data(file_path: Path) -> pd.DataFrame:
    """Charge le CSV et affiche un premier resume."""
    df = pd.read_csv(file_path, sep=";")
    print("=" * 80)
    print("1) IMPORTATION DES DONNEES")
    print("=" * 80)
    print(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("\nApercu:")
    print(df.head())
    print("\nTypes de colonnes:")
    print(df.dtypes)
    return df


def exploratory_analysis(df: pd.DataFrame) -> None:
    """Produit quelques graphiques exploratoires."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    print("\n" + "=" * 80)
    print("2) VISUALISATION EXPLORATOIRE")
    print("=" * 80)
    print("\nRepartition de la cible:")
    print(df[TARGET_COLUMN].value_counts())

    # 2.1 Distribution des classes
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Repartition des classes (Etat)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cannabis_class_balance.png", dpi=200)
    plt.close()

    # 2.2 Heatmap de correlations entre variables numeriques
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation entre variables chimiques")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cannabis_correlation_heatmap.png", dpi=200)
    plt.close()

    # 2.3 Boxplot de 4 variables a forte variance (choix simple)
    top4 = (
        df[numeric_cols]
        .var()
        .sort_values(ascending=False)
        .head(4)
        .index
        .tolist()
    )
    long_df = df[[TARGET_COLUMN] + top4].melt(
        id_vars=TARGET_COLUMN, var_name="Variable", value_name="Valeur"
    )
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=long_df, x="Variable", y="Valeur", hue=TARGET_COLUMN)
    plt.title("Comparaison de quelques marqueurs selon la classe")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cannabis_boxplots_top4.png", dpi=200)
    plt.close()

    print("Graphiques generes dans:", OUTPUT_DIR)


def run_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Projette les observations sur les 2 premieres composantes principales."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    X = df[numeric_cols]

    print("\n" + "=" * 80)
    print("3) PCA ET VISUALISATION")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df[TARGET_COLUMN] = df[TARGET_COLUMN].values
    pca_df["Var"] = df["Var"].values

    explained = pca.explained_variance_ratio_
    print(f"Variance expliquee PC1: {explained[0]:.3f}")
    print(f"Variance expliquee PC2: {explained[1]:.3f}")
    print(f"Variance expliquee cumulee: {explained.sum():.3f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=TARGET_COLUMN,
        style="Var",
        s=65,
        alpha=0.8,
    )
    plt.title("Projection PCA (PC1 vs PC2)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cannabis_pca_scatter.png", dpi=200)
    plt.close()

    print("Figure PCA generee dans:", OUTPUT_DIR)
    return pca_df


def run_classification(df: pd.DataFrame) -> None:
    """Entraine 2 modeles et compare leurs performances."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    X = df[numeric_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 80)
    print("4) CLASSIFICATION")
    print("=" * 80)

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, pos_label="Stupefiant")

        print("\n" + "-" * 80)
        print(f"Modele: {name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1 (classe 'Stupefiant'): {f1:.3f}")
        print("Classification report:")
        print(classification_report(y_test, pred))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # Matrice de confusion du meilleur modele
    assert best_model is not None
    best_pred = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Matrice de confusion - {best_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cannabis_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\nMeilleur modele (F1): {best_name} ({best_f1:.3f})")
    print("Matrice de confusion enregistree dans:", OUTPUT_DIR)


def main() -> None:
    sns.set_theme(style="whitegrid")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {DATA_PATH}. "
            "Lancez le script depuis la racine du projet SFC6006."
        )

    df = load_data(DATA_PATH)
    exploratory_analysis(df)
    _ = run_pca(df)
    run_classification(df)

    print("\nAnalyse terminee.")


if __name__ == "__main__":
    main()

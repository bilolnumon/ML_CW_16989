import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

st.set_page_config(layout="wide", page_title="Obesity ML App")


# Helper functions
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def show_df_info(df):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write(df.head())

def get_numeric_cats(df, target_col):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols, impute_strategy, encode_type, scale):
    transformers = []
    transformers.append(('num_imputer', SimpleImputer(strategy=impute_strategy), numeric_cols))
    if encode_type == "OneHot":
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        transformers.append(('cat_enc', ohe, categorical_cols))
    elif encode_type == "Label":
        transformers.append(('cat_passthrough', 'passthrough', categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    if scale:
        return preprocessor, StandardScaler()
    else:
        return preprocessor, None

def apply_label_encoding(df_train, df_val, df_test, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_val[col], df_test[col]], axis=0).astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_val[col] = le.transform(df_val[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))
        encoders[col] = le
    return df_train, df_val, df_test, encoders

def show_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    st.pyplot(plt.gcf())
    plt.close()

def detect_outliers_iqr(df, columns):
    outlier_indices = set()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices.update(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist())
    return outlier_indices


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data", "EDA", "Preprocessing", "Train & Tune", "Inference", "Evaluation"])


# App state
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'target' not in st.session_state:
    st.session_state.target = 'NObeyesdad'
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None


# Page: Data
if page == "Data":
    try:
        df = load_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
        st.session_state.df = df
        st.success("Loaded dataset from ObesityDataSet_raw_and_data_sinthetic.csv in this folder.")

        st.subheader("Original Dataset Preview")
        show_df_info(st.session_state.df)

        # --- Check for duplicates immediately ---
        dup_count = df.duplicated().sum()
        st.write(f"Number of duplicate rows in original dataset: {dup_count}")

        # --- Drop duplicates immediately ---
        if dup_count > 0:
            df = df.drop_duplicates()
            st.write(f"Dropped {dup_count} duplicate rows. New shape: {df.shape}")

        # Save cleaned df to session state for other pages
        st.session_state.df_cleaned = df

        # --- Round Age to nearest integer ---
        if 'Age' in df.columns:
            df['Age'] = df['Age'].round().astype(int)

        # Remove outliers
        numeric_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE']
        outlier_indices = detect_outliers_iqr(df, numeric_cols)
        df_cleaned = df.drop(index=outlier_indices)
     
        st.session_state.df_cleaned = df_cleaned
        st.success(f"Dataset loaded and cleaned from outliners. Removed {len(outlier_indices)} rows.")

    except FileNotFoundError:
        st.error("ObesityDataSet_raw_and_data_sinthetic.csv NOT FOUND in this folder.")
        st.stop()


    st.subheader("Cleaned Dataset Preview")
    show_df_info(st.session_state.df_cleaned)

    # Missing values
    missing_values = st.session_state.df_cleaned.isnull().sum()
    if missing_values.sum() > 0:
        st.subheader("Missing Values per Column")
        st.write(missing_values[missing_values > 0])
    else:
        st.write("No missing values found.")

# Page: EDA
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    df = st.session_state.df_cleaned

    if df is None:
        st.warning("Load dataset first on the Data page.")
    else:
        st.subheader("Statistical Summary")
        st.write(df.describe(include='all'))
        
        numeric_cols, categorical_cols = get_numeric_cats(df, st.session_state.target)

        st.subheader("Correlation Matrix (numeric features)")
        if len(numeric_cols) > 0:
            plt.figure(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.write("No numeric columns detected.")

        st.subheader("Histograms (Numeric Columns)")
        if numeric_cols:
            max_cols = min(12, len(numeric_cols))
            n = st.slider(
                "Max numeric columns to plot",
                min_value=1,
                max_value=max_cols,
                value=min(6, max(1, len(numeric_cols)))
            )
            plt.figure(figsize=(12, 6))
            df[numeric_cols[:n]].hist(figsize=(12, 6))
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.info("No numeric columns available for histograms.")

        st.subheader("Scatter: Height vs Weight (if present)")
        if 'Height' in df.columns and 'Weight' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x='Height',
                y='Weight',
                hue=st.session_state.target if st.session_state.target in df.columns else None,
                data=df
            )
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.info("Height/Weight not found in dataset.")

# Page: Preprocessing
elif page == "Preprocessing":
    st.header("Preprocessing choices")
    df = st.session_state.df_cleaned
    if df is None:
        st.warning("Load dataset first.")
    else:
        st.write("Detected columns:", df.columns.tolist())
        target = st.text_input("Target column name", value=st.session_state.target)
        st.session_state.target = target

        numeric_cols, categorical_cols = get_numeric_cats(df, target)
        st.write("Numeric columns:", numeric_cols)
        st.write("Categorical columns:", categorical_cols)

        impute_strategy = st.selectbox("Numeric imputation strategy", ["mean", "median"])
        encode_type = st.selectbox("Categorical encoding", ["OneHot", "Label"])
        scale_opt = st.checkbox("Apply Standard Scaling", True)

        if st.button("Apply preprocessing and split data"):
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            df_clean = df_clean.dropna(subset=[target])
            X = df_clean.drop(columns=[target])
            y = df_clean[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train if y_train.nunique()>1 else None)
            st.session_state.X_train_raw = X_train.copy()
            st.session_state.X_val_raw = X_val.copy()
            st.session_state.X_test_raw = X_test.copy()
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test

            preproc, scaler = build_preprocessor(numeric_cols, categorical_cols, impute_strategy, encode_type, scale_opt)
            
            if encode_type == "Label" and len(categorical_cols) > 0:
                X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
                X_val[categorical_cols] = X_val[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
                X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
                X_train, X_val, X_test, encoders = apply_label_encoding(
                    X_train, X_val, X_test, categorical_cols
                )
                imputer = SimpleImputer(strategy=impute_strategy)
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=numeric_cols + categorical_cols)
                X_val = pd.DataFrame(imputer.transform(X_val), columns=numeric_cols + categorical_cols)
                X_test = pd.DataFrame(imputer.transform(X_test), columns=numeric_cols + categorical_cols)
                if scale_opt:
                    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
                    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
                st.session_state.X_train = X_train
                st.session_state.X_val = X_val
                st.session_state.X_test = X_test
                st.session_state.preprocessor = ('label', encoders, scaler)
                st.success("Preprocessing (Label) applied and data split.")
            else:
                X_train_proc = preproc.fit_transform(X_train)
                X_val_proc = preproc.transform(X_val)
                X_test_proc = preproc.transform(X_test)
                if scaler is not None:
                    X_train_proc = scaler.fit_transform(X_train_proc)
                    X_val_proc = scaler.transform(X_val_proc)
                    X_test_proc = scaler.transform(X_test_proc)
                st.session_state.X_train = X_train_proc
                st.session_state.X_val = X_val_proc
                st.session_state.X_test = X_test_proc
                st.session_state.preprocessor = (preproc, scaler, numeric_cols, categorical_cols)
                st.success("Preprocessing applied and data split.")
            st.write("Train shape:", st.session_state.X_train.shape)
            st.write("Val shape:", st.session_state.X_val.shape)
            st.write("Test shape:", st.session_state.X_test.shape)


# Page: Train & Tune
elif page == "Train & Tune":
    st.header("Train models & hyperparameter tuning")
    if st.session_state.get('X_train') is None:
        st.warning("Run Preprocessing first.")
    else:
        X_train_proc = st.session_state.X_train
        X_val_proc = st.session_state.X_val
        X_test_proc = st.session_state.X_test
        y_train = st.session_state.y_train
        y_val = st.session_state.y_val
        y_test = st.session_state.y_test

        st.write("Choose models to train:")
        train_lr = st.checkbox("Logistic Regression", value=True, key="train_lr")
        train_rf = st.checkbox("Random Forest", value=True, key="train_rf")
        train_knn = st.checkbox("KNN", value=True, key="train_knn")
        tune = st.checkbox("Run hyperparameter tuning (GridSearchCV)", value=False, key="tune")

        if st.button("Train"):
            models = {}
            # Logistic Regression
            if train_lr:
                if tune:
                    lr_params = {"C":[0.01,0.1,1], "max_iter":[500]}
                    lr_grid = GridSearchCV(LogisticRegression(max_iter=500), lr_params, cv=3, scoring='accuracy', n_jobs=-1)
                    lr_grid.fit(X_train_proc, y_train)
                    models['lr'] = lr_grid.best_estimator_
                    st.write("LR best params:", lr_grid.best_params_)
                    st.session_state.lr_grid = lr_grid
                else:
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(X_train_proc, y_train)
                    models['lr'] = lr

            # Random Forest
            if train_rf:
                if tune:
                    rf_params = {"n_estimators":[100,200], "max_depth":[None,10]}
                    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
                    rf_grid.fit(X_train_proc, y_train)
                    models['rf'] = rf_grid.best_estimator_
                    st.write("RF best params:", rf_grid.best_params_)
                    st.session_state.rf_grid = rf_grid
                else:
                    rf = RandomForestClassifier(n_estimators=200, random_state=42)
                    rf.fit(X_train_proc, y_train)
                    models['rf'] = rf

            # KNN
            if train_knn:
                if tune:
                    knn_params = {"n_neighbors":[3,5,7]}
                    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='accuracy', n_jobs=-1)
                    knn_grid.fit(X_train_proc, y_train)
                    models['knn'] = knn_grid.best_estimator_
                    st.write("KNN best params:", knn_grid.best_params_)
                    st.session_state.knn_grid = knn_grid
                else:
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train_proc, y_train)
                    models['knn'] = knn

            st.session_state.models = models
            st.success("Models trained. Proceed to Inference page to see results.")


# Page: Inference
elif page == "Inference":
    st.header("Single-sample Inference")

    # SAFETY CHECKS
    if st.session_state.df is None:
        st.warning("Load dataset first (Data page).")
        st.stop()

    if not st.session_state.models:
        st.warning("Train models first (Train & Tune page).")
        st.stop()

    df = st.session_state.df
    target = st.session_state.target
    models = st.session_state.models
    feat_cols = [c for c in df.columns if c != target]

    st.subheader("Enter values for prediction")
    sample = {}

    # Input widgets
    for i, col in enumerate(feat_cols):
        widget_key = f"{col}_input_{i}"  # unique key

        if df[col].dtype in ['int64', 'float64']:
            if col.lower() == "age":
                sample[col] = st.slider("Age", 1, 100, int(df[col].median()), key=widget_key)
            elif col.lower() == "height":
                sample[col] = st.number_input("Height (m)", 0.5, 2.5, float(df[col].median()), key=widget_key)
            elif col.lower() == "weight":
                sample[col] = st.number_input("Weight (kg)", 10.0, 300.0, float(df[col].median()), key=widget_key)
            else:
                sample[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].median()), key=widget_key)
        else:
            options = df[col].dropna().unique().tolist()
            if len(options) <= 5:
                sample[col] = st.radio(col, options, key=widget_key)
            else:
                sample[col] = st.selectbox(col, options, key=widget_key)

    model_choice = st.selectbox("Choose model for prediction", list(models.keys()), key="model_select")

    # Predict button
    if st.button("Predict"):
        sample_df = pd.DataFrame([sample])
        preproc = st.session_state.preprocessor

        # Preprocessing
        if preproc[0] == 'label':  # Label encoding path
            encoders = preproc[1]
            scaler = preproc[2]

            for col, le in encoders.items():
                sample_df[col] = le.transform(sample_df[col].astype(str))

            imputer = SimpleImputer(strategy='mean')
            sample_df = pd.DataFrame(imputer.fit_transform(sample_df), columns=sample_df.columns)

            if scaler is not None:
                numeric_cols = sample_df.select_dtypes(include=['int64','float64']).columns
                sample_df[numeric_cols] = scaler.transform(sample_df[numeric_cols])

            X_in = sample_df.values

        else:  # ColumnTransformer path
            preproc_obj, scaler_obj, num_cols, cat_cols = preproc
            X_in = preproc_obj.transform(sample_df)
            if scaler_obj is not None:
                X_in = scaler_obj.transform(X_in)

        # Prediction
        model = models[model_choice]
        prediction = model.predict(X_in)[0]
        st.success(f"Prediction using {model_choice.upper()}: {prediction}")

        # BMI calculation
        if 'Height' in sample and 'Weight' in sample:
            height = float(sample['Height'])
            weight = float(sample['Weight'])
            if height > 0:
                bmi = weight / (height ** 2)
                st.info(f"📏 BMI: {bmi:.2f}")

                if bmi < 18.5:
                    st.write("Status: Underweight")
                elif bmi < 25:
                    st.write("Status: Normal weight")
                elif bmi < 30:
                    st.write("Status: Overweight")
                else:
                    st.write("Status: Obese")

# Page: Evaluation
elif page == "Evaluation":
    st.header("Evaluation & Comparison")
    if not st.session_state.models:
        st.warning("Train models first.")
    else:
        models = st.session_state.models
        X_test_proc = st.session_state.X_test
        y_test = st.session_state.y_test
        results = {}

        for name, model in models.items():
            y_pred = model.predict(X_test_proc)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            st.subheader(f"{name} - Accuracy: {acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,2))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Test Accuracy'])
        st.write("### Summary")
        st.dataframe(res_df)



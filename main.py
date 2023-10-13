import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVC
from datetime import datetime


def calculate_means(arr, n_split):
    i, j = n_split, n_split
    grid = np.array_split(arr, i, axis=1)
    grid = [np.array_split(row, j, axis=0) for row in grid]
    means = np.array([[np.mean(cell) for cell in row] for row in grid]).flatten()

    return means


def calculate_table_info(row):
    fluency_mapping = {
        'native': 1,
        'advanced': 2,
        'intermediate': 3,
        'basic': 4,
    }
    first_language_mapping = {
        'English (Canada)': 1,
        'Spanish (Venezuela)': 2,
        'English (United States)': 3,
        'Telugu': 4,
        'French (Canada)': 5
    }
    current_language_mapping = {
        'English (Australia)': 1,
        'Spanish (Venezuela)': 2,
        'English (United States)': 3,
        'English (Canada)': 4
    }
    gender_mapping = {
        'male': 1,
        'female': 2,
    }
    age_range_mapping = {
        '22-40': 1,
        '41-65': 2,
        '65+': 3
    }

    fluency = fluency_mapping.get(row['Self-reported fluency level '], 0)
    first_language = first_language_mapping.get(row['First Language spoken'], 0)
    current_language = current_language_mapping.get(row['Current language used for work/school'], 0)
    gender = gender_mapping.get(row['gender'], 0)
    age_range = age_range_mapping.get(row['ageRange'], 0)

    return fluency, first_language, current_language, gender, age_range


def load_model_data(row):
    y, sr = librosa.load(row['path'])
    y, _ = librosa.effects.trim(y, top_db=20)
    duration = librosa.get_duration(y=y, sr=sr)
    fluency, first_language, current_language, gender, age_range = calculate_table_info(row)
    if not ((age_range == 1 or age_range == 2) and
            (current_language == 1 or current_language == 3 or current_language == 4) and
            (fluency == 1 or fluency == 2) and
            (first_language == 1 or first_language == 3) and
            (0.3 <= duration <= 4)):
        return None
    rms = librosa.feature.rms(y=y)
    dbfs = librosa.power_to_db(rms)
    mean_dbfs = np.mean(dbfs)
    if mean_dbfs <= -50:
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    means = calculate_means(mel, 10)
    return means, row['action'] + row['object']


def load_evaluation_data(row):
    y, sr = librosa.load(row['path'])
    y, _ = librosa.effects.trim(y, top_db=20)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    means = calculate_means(mel, 10)
    row['mel'] = means
    return row


def get_result_random_forest(data_evaluation):
    # The hyperparameters to search over
    param_grid = {'n_estimators': [50, 100, 300, 500, 1000],
                  'max_depth': [5, 10, 15, 50, 100, None],
                  'criterion': ['gini', 'entropy']}

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0), n_jobs=-1)
    grid_search.fit(features, labels)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f'Best hyperparameters: {best_params}')
    print(f'Best score: {best_score:.2f}')

    # Save the results to a csv file
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('./rf_10.csv', index=False)

    best_random_forest = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                max_depth=best_params['max_depth'])
    best_random_forest.fit(features, labels)
    prediction = best_random_forest.predict(data_evaluation)
    e_df = pd.DataFrame({'Predicted': prediction})
    e_df.to_csv(path_or_buf='./rf_evaluation_10', sep=',', index_label='Id')


def get_result_svm(data_evaluation):
    # The hyperparameters to search over
    param_grid = {'C': [1, 10, 100],
                  'gamma': [0.1, 1, 'auto'],
                  }
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0), n_jobs=-1)
    grid_search.fit(features, labels)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best hyperparameters: {best_params}")
    print(f"Best score: {best_score:.2f}")

    # Save the results to a csv file
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv("./svm_10.csv", index=False)

    best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'])
    best_svm.fit(features, labels)
    prediction = best_svm.predict(data_evaluation)
    e_df = pd.DataFrame({'Predicted': prediction})
    e_df.to_csv(path_or_buf='./svm_evaluation_10', sep=',', index_label='Id')


print("start", datetime.now())
df = pd.read_csv("./development.csv")
df_eval = pd.read_csv("./evaluation.csv")

df_eval = df_eval.apply(load_evaluation_data, axis=1)
data_eval_feature = np.array(df_eval['mel'].apply(lambda x: x).tolist())

results = df.apply(load_model_data, axis=1).tolist()
labels = [r[1] for r in results if r is not None]
features = np.array([r[0] for r in results if r is not None])
print("loading over", datetime.now())

get_result_random_forest(data_evaluation=data_eval_feature)
print("random forest is over", datetime.now())

get_result_svm(data_evaluation=data_eval_feature)
print("svm is  over", datetime.now())

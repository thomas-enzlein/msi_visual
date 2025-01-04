import sys
import json
from msi_visual.normalization import total_ion_count, spatial_total_ion_count
from msi_visual.supervised.annotations import get_img, get_visualization, get_dataset
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import shap
import numpy as np
from msi_visual.extraction import get_extraction_mz_list
from scipy.stats import mannwhitneyu
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import tqdm
from sklearn.utils.class_weight import compute_sample_weight


if __name__ == "__main__":
    extraction_mzs = get_extraction_mz_list(r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\20_bins")

    paths = [(r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\20_bins\0.npy", 1),
            (r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\20_bins\2.npy", 0),
            (r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\20_bins\1.npy", 2),
            (r"E:\MSImaging-data\_msi_visual\Extractions\NRL4485-s2\20_bins\3.npy", 3)]


    similarity = {}
    separation_data = {}
    ks = {}

    X_all, y_all, label_encoder_all = get_dataset(r"NRL4485-s2_reannotation_23-12-24_PAHJ.json", paths, subsample=5, normalization=spatial_total_ion_count)
    categories = label_encoder_all.classes_
    for i in tqdm.tqdm(range(len(categories))):
        category1 = categories[i]
        for j in range(i+1, len(categories)):
            category2 = categories[j]
            a, b = label_encoder_all.transform([category1, category2])
            indices_a = [i for i in range(len(y_all)) if y_all[i] == a]
            indices_b = [i for i in range(len(y_all)) if y_all[i] == b]

            X_all_a = X_all[indices_a, :]
            X_all_b = X_all[indices_b, :]
            X = np.concatenate([X_all_a, X_all_b], axis=0)
            y = [0] * len(X_all_a) + [1] * len(X_all_b)    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            
            xgb_model = XGBClassifier(
                n_estimators=40,
                learning_rate=0.1,
                max_depth=4,
                random_state=42)
            
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = xgb_model.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, preds)
            print(category1, category2, X_train.shape, len(y_train), sample_weights.shape, np.mean(y_train), np.mean(y_test))
            print(category1, category2, balanced_accuracy, np.mean(y_test), np.mean(preds))
            print("*")

            similarity[(category1, category2)] = 1 - balanced_accuracy
            
            # # explainer = shap.TreeExplainer(xgb_model)
            # # shap_values = explainer.shap_values(X_test[::5, :])
            # # meanabs = np.abs(shap_values).mean(axis=0)
            
            # # indices = list(np.argsort(meanabs)[-20 : ])

            indices = list(range(len(extraction_mzs)))

            separation_data[(category1, category2)] = []
            separation_data[(category2, category1)] = []



            for index in indices:
                x_a, x_b = X_all_a[:, index], X_all_b[:, index]
                statistic, pvalue = mannwhitneyu(x_a, x_b)
                statistic = statistic / (len(x_a) * len(x_b))
                intensity1 = np.mean(x_a)
                intensity2 = np.mean(x_b)
                if pvalue < 0.05 and statistic > 0.6:
                    separation_data[(category1, category2)].append((extraction_mzs[index], statistic, pvalue, intensity1, intensity2))
                if pvalue < 0.05 and statistic < 0.4:
                    separation_data[(category2, category1)].append((extraction_mzs[index], 1-statistic, pvalue, intensity2, intensity1))

            print(category1, category2, len(separation_data[(category1, category2)]), len(separation_data[(category2, category1)]))

            separation_data[(category1, category2)] = sorted(separation_data[(category1, category2)], key=lambda x: x[1], reverse=True)
            separation_data[(category2, category1)] = sorted(separation_data[(category2, category1)], key=lambda x: x[1], reverse=True)

                
        result = {"ks": ks, "separation_data": separation_data, "similarity": similarity, "categories": categories}

        with open("graph_data_coarse_model_20bin_27-12-2024.joblib", "wb") as f:
            joblib.dump(result, f)
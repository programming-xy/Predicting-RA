# -------------------
# 模型配置------分离目标污染物基本配置信息
# -------------------
MODEL_CONFIG = {
    "Pb": {
        "model_path": r'Fin_model\best_xgb_Fin_Pb.pkl',
        "feature_path": r'Features_list\feature_names_Fin_Pb.txt',
        "train_data_path": r'example_data\X_train_Pb.csv',
        "target_name": "Ratio of Activation of Soil Pb",
        "model_type": "XGBoost"
    },
    "Cd": {
        "model_path": r'Fin_model\best_xgb_Fin_Cd.pkl',
        "feature_path": r'Features_list\feature_names_Fin_Cd.txt',
        "train_data_path": r'example_data\X_train_Cd.csv',
        "target_name": "Ratio of Activation of Soil Cd",
        "model_type": "XGBoost"
    },
    "As": {
        "model_path": r'Fin_model\best_cb_Fin_As.pkl',
        "feature_path": r'Features_list\feature_names_Fin_As.txt',
        "train_data_path": r'example_data\X_train_As.csv',
        "target_name": "Ratio of Activation of Soil As",
        "model_type": "CatBoost"
    }
}

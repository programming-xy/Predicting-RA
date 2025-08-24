def load_model_and_data(model_config):
    # 1. 手动指定编码（根据 VS Code 检测结果填，比如 'utf-8'/'gbk'）
    ENCODING = 'utf-8'  # 替换成你实际的编码！！！
    
    try:
        # 2. 加载模型（不变）
        model = joblib.load(model_config["model_path"])
    except Exception as e:
        raise ValueError(f"模型加载失败：{str(e)}")

    try:
        # 3. 读取特征文件（用手动指定的编码）
        with open(model_config["feature_path"], 'r', encoding=ENCODING) as f:
            feature_names = [line.strip() for line in f if line.strip()]
        
        # 4. 修复已知乱码（比如 'ï¿¥X' 是 'ΔX' 的错误解码）
        #    这里需要你替换成实际乱码和正确值！！！
        feature_names = [
            'ΔX' if name == 'ï¿¥X' else name  
            for name in feature_names
        ]

        # 5. 读取 CSV 并校验特征
        X_train = pd.read_csv(model_config["train_data_path"])
        missing_features = [f for f in feature_names if f not in X_train.columns]
        if missing_features:
            raise ValueError(f"CSV 缺少特征：{missing_features}")
        X_train = X_train[feature_names]

        return model, feature_names, X_train
    
    except Exception as e:
        raise ValueError(f"数据加载失败：{str(e)}")

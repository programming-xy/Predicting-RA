# --------------------
    # 定义方法1------分离模型（特征名称）与数据加载函数
    # --------------------
    import joblib
    import pandas as pd
    import chardet  # 用于检测文件编码

    def load_model_and_data(model_config):
        """加载模型、特征名称和训练数据，修复乱码问题"""
        try:
            # 1. 加载模型
            model = joblib.load(model_config["model_path"])
        except Exception as e:
            raise ValueError(f"模型加载失败：{str(e)}")

        try:
            # 2. 读取特征文件并检测编码
            with open(model_config["feature_path"], 'rb') as f:
                raw_data = f.read()
                # 检测文件编码
                detected_encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                # 使用检测到的编码解码，无法解码时替换异常字符
                content = raw_data.decode(detected_encoding, errors='replace')
            
            # 处理特征名称（清理空行和空白字符）
            feature_names = [line.strip() for line in content.splitlines() if line.strip()]

            # 关键修改：精准替换乱码为ΔX
            # 注意：根据实际报错的乱码调整替换字符串
            feature_names = [
                name.replace('[Î”X]', 'ΔX')    # 替换带括号的乱码
                   .replace('Î”X', 'ΔX')      # 替换不带括号的乱码
                   .replace('ï¿¥X', 'ΔX')     # 常见的ΔX乱码形式
                for name in feature_names
            ]
            
            # 调试输出：查看替换后的特征名（部署时可注释）
            print("替换后的特征名列表：", feature_names)
            
            # 3. 读取训练数据并验证特征列
            X_train = pd.read_csv(model_config["train_data_path"])
            
            # 检查特征名是否都在CSV列中
            missing_features = [f for f in feature_names if f not in X_train.columns]
            if missing_features:
                # 额外提示：打印CSV中实际存在的列名，方便对比
                print("CSV文件中的实际列名：", X_train.columns.tolist())
                raise ValueError(f"CSV文件中缺少以下特征列：{missing_features}")
            
            # 选取有效特征列
            X_train = X_train[feature_names]
            
            return model, feature_names, X_train
        
        except Exception as e:
            raise ValueError(f"数据加载失败：{str(e)}")

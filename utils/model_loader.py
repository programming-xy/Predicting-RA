# -------------------
# 定义方法1------分离模型（特征名称）与数据加载函数
# -------------------
import joblib
import pandas as pd

def load_model_and_data(model_config):
    """加载模型、特征名称和训练数据示例"""
    model = joblib.load(model_config["model_path"])
    
    with open(model_config["feature_path"], 'rb') as f:
        content = f.read()
    
    # 强制转成字符串：先用 Latin-1 过渡（不会报错），再替换乱码
    feature_names = content.decode('latin-1', errors='ignore').splitlines()
    feature_names = [line.strip() for line in feature_names if line.strip()]  # 清理空行
    
    X_train = pd.read_csv(model_config["train_data_path"])[feature_names]
    
    return model, feature_names, X_train

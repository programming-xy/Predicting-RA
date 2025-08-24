# --------------------
# 定义方法1------分离模型（特征名称）与数据加载函数
# --------------------
import joblib
import pandas as pd
import chardet  # 需安装：pip install chardet

def load_model_and_data(model_config):
    """加载模型、特征名称（修复ΔX乱码）"""
    # 1. 加载模型（二进制模式，无编码问题）
    try:
        model = joblib.load(model_config["model_path"])
    except Exception as e:
        raise ValueError(f"模型加载失败：{str(e)}")

    # 2. 读取特征文件并智能处理编码
    try:
        with open(model_config["feature_path"], 'rb') as f:
            raw_data = f.read()
            
            # 自动检测编码（优先用文件真实编码）
            detected_encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            # 解码：无法识别的字符替换为“�”，便于定位乱码
            content = raw_data.decode(detected_encoding, errors='replace')
            
            # 清理特征名（去空行、去空白）
            feature_names = [line.strip() for line in content.splitlines() if line.strip()]
            
            # 3. 强制修复ΔX乱码（根据实际乱码调整替换规则）
            #    示例：常见乱码形式（需根据报错调整）
            feature_names = [
                name.replace('î”X', 'ΔX')  # 匹配报错中的乱码 [î\x94X]
                   .replace('ï¿¥X', 'ΔX')     # 其他常见乱码形式
                   .replace('&#916;X', 'ΔX')  # HTML实体编码
                for name in feature_names
            ]

            # 4. 验证特征与 CSV 列名匹配
            X_train = pd.read_csv(model_config["train_data_path"])
            missing_features = [f for f in feature_names if f not in X_train.columns]
            
            if missing_features:
                # 调试：打印 CSV 实际列名，辅助排查
                print(f"CSV 实际列名：{X_train.columns.tolist()}")
                raise ValueError(f"CSV 缺少特征：{missing_features}")
            
            # 选取有效特征列
            X_train = X_train[feature_names]

            return model, feature_names, X_train
        
    except Exception as e:
        raise ValueError(f"数据加载失败：{str(e)}")

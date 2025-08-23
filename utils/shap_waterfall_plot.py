# -------------------
# 定义方法2------分离shap瀑布图绘制函数
# -------------------
import shap
import matplotlib.pyplot as plt

def create_waterfall_plot(model, X_train, sample, feature_names, metal_name, max_display=10):
    """创建SHAP瀑布图"""
    explainer = shap.TreeExplainer(model, data=X_train)  #使用示例数据初始化shap解释器
    shap_values = explainer.shap_values(sample)          #计算预测样本shap值
    
    explanation = shap.Explanation(                      #创建shap解释对象
        values=shap_values[0],                           #提取预测样本shap值
        base_values=explainer.expected_value,            #提取解释器基准值
        data=sample.iloc[0],                             #提取预测样本特征值
        feature_names=feature_names                      #特征清单，便于解读shap值
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)   #绘制shap瀑布图
    
    plt.title(f"{metal_name} Feature Contribution Waterfall Plot", fontsize=14)
    plt.xlabel("Prediction Deviation", fontsize=12)
    plt.tight_layout()
    
    return fig
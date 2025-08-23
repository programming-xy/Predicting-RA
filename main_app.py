# -------------------
# 主程序--------仅保留主程序底层代码运行逻辑，结构更加清晰！
# -------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from datetime import datetime

from config.model_config import MODEL_CONFIG
from utils.model_loader import load_model_and_data
from utils.shap_waterfall_plot import create_waterfall_plot

# 设置字体
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# Streamlit界面
#st.title("耦合SHAP算法与机器学习模型的土壤重金属激活比率预测系统 📊")
st.title("Prediction System for Ratio of Activation of Soil Heavy Metals Integrating SHAP Algorithm and Machine Learning Model📊")
#st.markdown("### 🛠️ 多金属多模型交互式预测与SHAP特征解释")
st.markdown("### 🛠️ Interactive Prediction of Multi-Metals with Multiple Models and SHAP Feature Interpretation")

# 选择金属模型
#metal_selection = st.selectbox(
  #  "重金属类型选择",
    #list(MODEL_CONFIG.keys())
#)
metal_selection = st.selectbox(
    "Type Selection of Heavy metals",
    list(MODEL_CONFIG.keys())
)

# 获取所选模型的配置
model_config = MODEL_CONFIG[metal_selection]

# 加载模型和数据
with st.spinner(f"正在加载{metal_selection}模型和数据..."):
    try:
        model, feature_names, X_train = load_model_and_data(model_config)
        #st.success(f"✅ {metal_selection}模型加载成功，共{len(feature_names)}个特征")
        st.success(f"✅ The model for {metal_selection} was loaded successfully, and there are a total of {len(feature_names)} features.")
    except Exception as e:
        #st.error(f"❌ 模型加载失败: {str(e)}")
        st.error(f"❌ Failed to load the model. Please ensure the model file exists and is in the correct format: {str(e)}")
        st.stop()

# 为用户提供一组预测样本示例
example_data = {
    "Pb": {
        'ΔX': 0.036711,
        'TPb': 21.43,
        'APb': 1.87,
        'AAs': 0.058537,
        'TCr': 89.05,
        'SM': 3.00,
        'PSN': 30.64,
        'CEC': 11.64,
        'TA': 1.94
    },
    "Cd": {
        'Dist': 0.163, 
        'TCd': 0.034, 
        'ACd': 0.016, 
        'TCr': 24.79, 
        'pH': 6.11, 
        'CEC': 3.75
    },
    "As": {
        'lat': 19.22, 
        'TAs': 25.99, 
        'AAs':0.076, 
        'TCr': 22.62, 
        'TT': 70.945300
    }
}

# 初始化会话状态
if 'input_values' not in st.session_state:
    st.session_state.input_values = example_data[metal_selection].copy()    #判别是否使用系统提供预测样本

# 使用预测样本示例按钮
#if st.button(f"使用{metal_selection}预测样本示例"): 
if st.button(f"Use Exemple Sample for {metal_selection} Prediction"):
    st.session_state.input_values = example_data[metal_selection].copy()

# 创建特征输入表单
with st.form("feature_form"):
    #st.subheader(f"输入{metal_selection}特征值")      #为用户提供用于系统预测的特征值输入交互交互界面
    st.subheader(f"Enter {metal_selection} Feature Values")

    # 创建两列布局
    cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        # 获取统计信息
        min_val = X_train[feature].min()
        max_val = X_train[feature].max()
        mean_val = X_train[feature].mean()
        std_val = X_train[feature].std()
        
        # 默认值使用均值
        default_value = st.session_state.input_values.get(feature, mean_val)     #预测系统启动后，每种金属特征值输入对话框默认使用系统提供训练数据示例数据
        
        # 在两列中交替显示输入框
        col = cols[i % 2]
        
        # 创建数字输入框
        value = col.number_input(
            f"{feature}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_value),
            step=0.001,
            format="%.3f",
            help=f"Range: {min_val:.3f} - {max_val:.3f} (Mean: {mean_val:.3f}, Std Dev: {std_val:.3f})"
        )
        
        st.session_state.input_values[feature] = value
    
    # 瀑布图设置
    #st.subheader("瀑布图设置")
    st.subheader("Waterfall Plot Settings")
    #max_display = st.slider("显示的特征数量", 3, 20, 10)    #用户可自定义特征显示数量
    max_display = st.slider("Number of Features to Display", 3, 20, 10)
    # 执行预测
    submitted = st.form_submit_button(f"🚀 Execute {metal_selection} Prediction")

# 处理预测请求
if submitted:
    with st.spinner(f"💫 Calculating {metal_selection} prediction results and SHAP interpretation..."):
        input_data = pd.DataFrame([st.session_state.input_values])[feature_names]
        prediction = model.predict(input_data)[0]
        
        st.subheader("📊 Prediction Result")
        st.metric(label=model_config["target_name"], value=f"{prediction:.4f}")
        
        # 生成瀑布图
        st.subheader(f"🔍 {metal_selection} Feature Contribution Waterfall Plot")
        try:
            waterfall_fig = create_waterfall_plot(
                model, X_train, input_data, feature_names, metal_selection,
                max_display=max_display
            )
            st.pyplot(waterfall_fig)
        except Exception as e:
            st.error(f"❌ Failed to generate waterfall plot: {str(e)}")
        
        # 显示原始输入
        with st.expander("View Input Details"):
            st.dataframe(input_data)
        
        # 显示基准值
        if model_config["model_type"] in ["XGBoost", "CatBoost"]:
            explainer = shap.TreeExplainer(model, data=X_train)
            base_value = explainer.expected_value
            st.info(f"Model Baseline Value: {base_value:.4f}")

        # 显示预测时间-------预测可溯
        st.info(f"Prediction Time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


#开发者CMD启动命令：D:-cd "D:\硕士科研\Programming\github"-.\venv\Scripts\activate.bat（'-'是分割启动命令的每个步骤分隔符）
#streamlit run "D:\硕士科研\Programming\github\main_app.py"
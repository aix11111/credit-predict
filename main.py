In[1]:# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

riskdat = pd.read_csv('train.csv')
# 数据的维度: 36000个样本, 13个自变量, 1个因变量
riskdat.shape
Out[1]:(36000, 14)


In[2]:#查看数据
riskdat.head()

Out[2]:个人年龄	个人性别	个人教育程度	个人收入	个人工作经验	个人房屋所有权	借记卡余额	信用卡使用意向	信用卡利率	贷款占收入的百分比	个人信用历史时长	信用评分	档案中以往贷款违约情况	信用卡状态
0	24	male	Master	145248	1	RENT	14000	MEDICAL	10.00	0.10	3	573	Yes	0
1	35	male	Bachelor	88152	12	RENT	15000	HOMEIMPROVEMENT	15.57	0.17	8	553	No	1
2	53	male	Bachelor	128167	28	MORTGAGE	20000	PERSONAL	9.88	0.16	23	626	Yes	0
3	22	male	Master	68078	1	MORTGAGE	-2222222000	DEBTCONSOLIDATION	7.51	0.03	4	632	Yes	0
4	33	male	Bachelor	342947	12	MORTGAGE	10000	DEBTCONSOLIDATION	13.99	0.03	7	671	No	0

In[3]:#保留贷款违约情况
riskdat['档案中以往贷款违约情况'] = riskdat['档案中以往贷款违约情况'].map({'yes':1, 'no':0})
riskdat.head()
Out[3]:个人年龄	个人性别	个人教育程度	个人收入	个人工作经验	个人房屋所有权	借记卡余额	信用卡使用意向	信用卡利率	贷款占收入的百分比	个人信用历史时长	信用评分	档案中以往贷款违约情况	信用卡状态
0	24	male	Master	145248	1	RENT	14000	MEDICAL	10.00	0.10	3	573	1	0
1	35	male	Bachelor	88152	12	RENT	15000	HOMEIMPROVEMENT	15.57	0.17	8	553	0	1
2	53	male	Bachelor	128167	28	MORTGAGE	20000	PERSONAL	9.88	0.16	23	626	1	0
3	22	male	Master	68078	1	MORTGAGE	-2222222000	DEBTCONSOLIDATION	7.51	0.03	4	632	1	0
4	33	male	Bachelor	342947	12	MORTGAGE	10000	DEBTCONSOLIDATION	13.99	0.03	7	671	0	0


In[4]:# 删除全 0 列
riskdat = riskdat.loc[:, ~((riskdat == 0).all(axis=0))]
Out[4]:	个人年龄	个人性别	个人教育程度	个人收入	个人工作经验	个人房屋所有权	借记卡余额	信用卡使用意向	信用卡利率	贷款占收入的百分比	个人信用历史时长	信用评分	档案中以往贷款违约情况	信用卡状态
0	24	male	Master	145248	1	RENT	14000	MEDICAL	10.00	0.10	3	573	1	0
1	35	male	Bachelor	88152	12	RENT	15000	HOMEIMPROVEMENT	15.57	0.17	8	553	0	1
2	53	male	Bachelor	128167	28	MORTGAGE	20000	PERSONAL	9.88	0.16	23	626	1	0
3	22	male	Master	68078	1	MORTGAGE	-2222222000	DEBTCONSOLIDATION	7.51	0.03	4	632	1	0
4	33	male	Bachelor	342947	12	MORTGAGE	10000	DEBTCONSOLIDATION	13.99	0.03	7	671	0	0

In[5]:#分离数值型特征和非数值型特征
riskdat_sub1 = riskdat.select_dtypes(exclude=['object'])

In[6]:# 用均值填充缺失的数值型变量
riskdat_sub1 = riskdat_sub1.fillna(riskdat_sub1.mean())
riskdat.head()
Out[6]:个人年龄	个人性别	个人教育程度	个人收入	个人工作经验	个人房屋所有权	借记卡余额	信用卡使用意向	信用卡利率	贷款占收入的百分比	个人信用历史时长	信用评分	档案中以往贷款违约情况	信用卡状态
0	24	male	Master	145248	1	RENT	14000	MEDICAL	10.00	0.10	3	573	1	0
1	35	male	Bachelor	88152	12	RENT	15000	HOMEIMPROVEMENT	15.57	0.17	8	553	0	1
2	53	male	Bachelor	128167	28	MORTGAGE	20000	PERSONAL	9.88	0.16	23	626	1	0
3	22	male	Master	68078	1	MORTGAGE	-2222222000	DEBTCONSOLIDATION	7.51	0.03	4	632	1	0
4	33	male	Bachelor	342947	12	MORTGAGE	10000	DEBTCONSOLIDATION	13.99	0.03	7	671	0	0

In[7]:
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler


In[8]:# 提取 X 和 y
target_column = '信用卡状态'
X = riskdat_sub1.iloc[:,:-1]
y = riskdat_sub1[target_column]
riskdat.shape

In[9]:# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=114)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
riskdat.head()


In[10]:
# XGBoost 模型
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=114,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1 # 类别不平衡处理
)


# In[151]:


# 超参数范围
param_grid = {
    'n_estimators': [100,200,300],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'max_depth': [3,5,7],
    'subsample': [0.7,0.8,0.9],
    'colsample_bytree': [0.7,0.8,0.9],
    'scale_pos_weight': [1,5,10]
}


# In[152]:


# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_xgb_clf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


# In[153]:


# 训练集预测并评估
y_train_pred = best_xgb_clf.predict(X_train)
print("Confusion matrix (training):\n", confusion_matrix(y_train, y_train_pred))
print("Classification report (training):\n", classification_report(y_train, y_train_pred))


# In[154]:


# 验证集预测并评估
y_val_pred = best_xgb_clf.predict(X_val)
y_val_pred_prob = best_xgb_clf.predict_proba(X_val)[:, 1]
print("Confusion matrix (validation):\n", confusion_matrix(y_val, y_val_pred))
print("Classification report (validation):\n", classification_report(y_val, y_val_pred))


# In[155]:


# ROC 曲线绘制
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_val_pred_prob)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)


plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='XGBoost (AUC = {:.2f})'.format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


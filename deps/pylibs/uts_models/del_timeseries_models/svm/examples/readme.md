## 在4个数据集上的异常检测效果图
每个图分为上下两行:
- 上: 原始时间序列数据(蓝色正常, 红色异常)
- 下: 蓝色: score, 红色虚线: 取到 best f1 score 时对于的阈值 
- Rules for normal and anomaly:  anomaly if score >= threshold, otherwise normal. 
![05_svm_a.png](05_svm_a.png)
![01_svm_l.png](01_svm_l.png)
![02_svm_l.png](02_svm_l.png)
![03_svm_c.png](03_svm_c.png)
![04_svm_c.png](04_svm_c.png)
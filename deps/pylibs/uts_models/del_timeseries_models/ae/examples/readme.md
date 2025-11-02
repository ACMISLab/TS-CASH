## 在4个数据集上的异常检测效果图
每个图分为上下两行:
- 上: 原始时间序列数据(蓝色正常, 红色异常)
- 下: 蓝色: score, 红色虚线: 取到 best f1 score 时对于的阈值 
- Rules for normal and anomaly:  anomaly if score >= threshold, otherwise normal. 
![01_ae_pe.png](01_ae_pe.png)
![02_ae_pe.png](02_ae_pe.png)
![03_ae_fl.png](03_ae_fl.png)
![04_ae_fl.png](04_ae_fl.png)
![05_ae_ai.png](05_ae_ai.png)
![06_ae_ai.png](06_ae_ai.png)
![07_ae_ai.png](07_ae_ai.png)
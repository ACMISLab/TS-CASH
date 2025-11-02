## 在4个数据集上的异常检测效果图
每个图分为上下两行:
- 上: 原始时间序列数据(蓝色正常, 红色异常)
- 下: 蓝色: score, 红色虚线: 取到 best f1 score 时对于的阈值 
- Rules for normal and anomaly:  anomaly if score >= threshold, otherwise normal. 
![fake_kpi_01_vague.png](fake_kpi_01_vague.png)
![fake_kpi_01.png](fake_kpi_01.png)
![fake_period_obvious.png](fake_period_obvious.png)
![fake_period_vague.png](fake_period_vague.png)
![e0747cad-8dc8-38a9-a9ab-855b61f5551d.png](e0747cad-8dc8-38a9-a9ab-855b61f5551d.png)
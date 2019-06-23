# 航空公司客户价值分析

标签： 聚类分析

---

## 1.背景和目标
背景：

通过客户分类，可区分无价值客户和高价值客户，企业针对不同价值客户定制个性化服务方案，可将有限资源集中高价值客户，实现企业利润最大化。

目标：

* 通过数据对客户进行分类
* 对不同客户进行特征分析，比较不同类客户的价值
* 对不同价值客户进行个性化服务，制定相应的营销策略

----------
## 2.分析方法确定
* 传统方法为使用RFM模型衡量客户价值，而在M（monetary消费金额）中，因航空票价同时由季度、购买舱位等级决定，因此在本案例中，应使用客户的飞行里程M与所乘坐舱位对应的平均折扣C率代替
* 用户的入会时长L也能影响客户价值，因此模型为LRFMC模型
* 若使用传统的RFM属性分箱分析方法，客户群将有2^5个，增加了个性化营销成本，因而使用K-means聚类分析方法

----------
## 3.分析过程

```flow
st=>start: 获取数据
io1=>inputoutput: 对数据做探索分析和预处理
io2=>inputoutput: 对已预处理数据建模
op=>operation: 针对不同价值客户进行个性化营销服务
e=>end: End

st->io1->io2->op->e
```
### 3.1数据抽取及预处理
* 按LRFMC模型从数据集取出相关属性数据：'FFP_DATE','LOAD_TIME','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount'
* 数据属性说明：
FFP_DATE：入会时间（办理会员卡的开始的时间）
LOAD_TIME：观测窗口的结束时间（选取样本的时间宽度，距离现在最近的时间）
FLIGHT_COUNT：飞行次数（频数）
SEG_KM_SUM：观测窗口总飞行公里数	
LAST_TO_END：最后一次乘机时间至观察窗口末端时长	
avg_discount：平均折扣率

### 3.2数据预处理
* 分析是否有空值，若有空值数据较少，针对现有样本可对有空值数据进行剔除
* 'FFP_DATE','LOAD_TIME'属性均为object，将其进行转换得到指标
L=FFP_DATE-LOAD_TIME
其他属性则对应指标
R=LAST_TO_END
F=FLIGHT_COUNT
M=SEG_KM_SUM
C=AVG_DISCOUNT
* 对各属性数据做归一化


----------
## 4.数据建模

* 因通过手肘法判断K值时图像显示不明显

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline

SSE=[]
for k in range(1,9):
    pred=KMeans(n_clusters=k).fit(data_clean_norm)
    SSE.append(pred.inertia_)
x=range(1,9)
plt.plot(x,SSE,'o-')
plt.ylabel('SSE')
plt.xlabel('x')
```

* 故使用Gap Statistic方法计算得出最佳K值  ，但由于类别较多带来营销成本上升，故按照“重点保持客户”、“重点发展客户”、“重点挽留客户”、“低价值客户”4类进行分析
```python
import numpy as np
from sklearn.cluster import KMeans

def gap(data,nrefs=20,maxclusters=15):
    gaps=np.zeros(len(range(maxclusters+1)))
    resultdf=pd.DataFrame({'clusterCount':[],'gap':[]})
    for gap_index,k in enumerate(range(1,maxclusters+1)):
        refs=np.zeros(nrefs)
        for i in range(nrefs):
            pred=KMeans(n_clusters=k,init='k-means++').fit(data)
            refs[i]=pred.inertia_
        pred_once=KMeans(n_clusters=k,init='k-means++').fit(data)
        refs_once=pred_once.inertia_
        gap=np.mean(np.log(refs))-refs_once
        resultdf.append({'clusterCount':k,'gap':gap},ignore_index=True)
    return resultdf['gap'].values.argmax()+1,resultdf
gap(data_clean_norm,maxclusters=15)
```

* 4类客户特征如下：
重点保持客户：R小，F、M、C大，是价值最高的客户
重点发展客户：R、F、M、L小，但C大，是公司的潜在价值客户
重点挽留客户：C、L、R大，F、M小
低价值客户：R大，L、F、M、C小，一般只在机票打折时才购买机票


----------
## 5.对应方案

* 针对重点保持客户，可采用里程兑换机票或升舱，并及时在里程达到相应阈值时提醒客户
* 针对重点发展客户，通过定期告知客户目前积累的里程，若乘机积累里程，可以升级会员卡，从而享受更大的会员福利
* 针对重点挽留客户，可与其他企业合作，采用交叉销售的方法，让此类客户在本公司合作伙伴上消费时对机票进行相应的折扣以吸引客户

----------
##6.小结
针对传统FRM模型的不足，采用K-Means算法进行分析，挖掘出4类用户，针对4类用户进行不同的营销方案。
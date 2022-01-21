# LCG T-TEST USING EUCLIDEAN METHOD
-----------------------------------
## Advanced Analytics and Growth Marketing Telkomsel
----------------------------------------------------
- Project Supervisor : Rizli Anshari, General Manager of AAGM Telkomsel
- Writer             : Azka Rohbiya Ramadani, Muhammad Gilang, Demi Lazuardi

This project has been created for statistical usage, purposing for determining ATL takers and nontakers using LCG ttest and Euclidean Method, especially for internal business case in Telkomsel.

## Background
-------------
In offering digital product, bussiness analyst must have considered what is the most suitable criterias of customers who have potential to buy the product. As an illustration, targetting gamers for games product campaign is better decision rather than targetting random customers without knowing customers behaviour. However, bussiness wouldn't make all the gamers as the campaign target, otherwise, market team deliberately wouldn't offer to several gamers randomly as comparison, called as Control Group. Therefore, it enables the team in measuring how success the campaign is.

After the campaign, there are product takers who are expected comes from campaign target. Additionaly, nontakers group is expected comes from Control Group and non-target customers. The analysis problems emerge afterwards, since nontakers might come from outside target. For this reason, our team made statistical technique in python algorithm to determine Control Group by applying euclideanmethod-combined t-test, in order for comparing takers and nontakers behaviour before the campaign, in this case we're focusing on revenue before. As a result, it enables bussiness analyst evaluating how success the campaign is.

## Installation Guide
---------------------
This algorith has been uploaded to pypi.org. Therefore, in order to get package, you can easely download using the following command

`pip3 install lcgeuclideanmethod`

## Requirements
---------------
This python requires related package
more importantly python_requires='>=3.1', so that package can be install
Make sure the other packages meet the requirements below
- pandas>=1.1.5,
- numpy>=1.18.5,
- scipy>=1.2.0,
- matplotlib>=3.1.0,
- statsmodels>=0.8.0

## Usage Guide
--------------
### 1. EuclideanMethod
- Input:
  - df_takers     : dataframe takers containing two columns, customers and revenue before campaign
  - df_nontakers  : dataframe nontakers containing two columns, customers and revenue before campaign
- Output:
  - summary       : containing the number of expected Control Group populations based on max-p value, and other general information like average, std, max, min, etc.
  - df_result     : subprocess table to find p-value from random nontakers
  - df_tukey      : main result containing category customers category, based on summary calculation
  - tukey         : tukey HSD evaluation, readmore [Tukey HSD](https://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html)

sample code
```
from lcgttest.lcgttest import EuclideanMethod
import pandas as pd

# where you put takers and nontakers file
df_takers = pd.read_csv('takers.csv')
df_nontakers = pd.read_csv('nontakers.csv')

model = EuclideanMethod(df_takers, df_nontakers)
model.run()

# output
print(model.summary)
print(model.df_result)
print(model.df_tukey)
print(model.tukey)
```
### 2. MapEuclideanMethod
This is like map function in python
- Input:
  - arr_df_takers     : dataframe takers but in array form
  - arr_df_nontakers  : dataframe nontakers but in array form
  - labels            : labels of both takers and nontakers in array form
- Output:
  - df_summary       : containing the number of expected Control Group populations based on max-p value, and other general information like average, std, max, min, etc in dataframe form.
  - dict_df_result     : subprocess table to find p-value from random nontakers in dicttionary type.
  - dict_df_tukey      : main result containing category customers category, based on summary calculation in dicttionary type.
  - dict_tukey         : tukey HSD evaluation, readmore [Tukey HSD](https://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html) in dicttionary type.

sample code
```
from lcgttest.lcgttest import MapEuclideanMethod
import pandas as pd
import numpy as np

# where you put takers and nontakers file
arr_df_takers = np.array([df_takers, df_takers2, df_takers3])
arr_df_takers = np.array([df_nontakers, df_nontakers2, df_nontakers3])
labels = ['campaignA','campaignB','campaignC']

model2 = MapEuclideanMethod(arr_df_takers, arr_df_nontakers, label = labels )

# output
print(model.df_summary)
print(model.dict_df_result)
print(model.dict_df_tukey)
print(model.dict_tukey)
```

### 3. EuclideanMethodAscDesc
This is run twice MapEuclideanMethod ascending and descending (technique to randomize the nontakers samples)
- Input:
  - arr_df_takers     : dataframe takers but in array form
  - arr_df_nontakers  : dataframe nontakers but in array form
  - labels            : labels of both takers and nontakers in array form
- Output:
  - df_summary       : containing the number of expected Control Group populations based on max-p value, and other general information like average, std, max, min, etc in dataframe form.
  - dict_df_result     : subprocess table to find p-value from random nontakers in dicttionary type.
  - dict_df_tukey      : main result containing category customers category, based on summary calculation in dicttionary type.
  - dict_tukey         : tukey HSD evaluation, readmore [Tukey HSD](https://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html) in dicttionary type.

sample code
```
from lcgttest.lcgttest import EuclideanMethodAscDesc
import pandas as pd
import numpy as np

# where you put takers and nontakers file
arr_df_takers = np.array([df_takers, df_takers2, df_takers3])
arr_df_takers = np.array([df_nontakers, df_nontakers2, df_nontakers3])
labels = ['campaignA','campaignB','campaignC']

model3 = EuclideanMethodAscDesc(arr_df_takers, arr_df_nontakers, label = labels )

# output
print(model3.df_summary)
print(model3.dict_df_result)
print(model3.dict_df_tukey)
print(model3.dict_tukey)

# additional input
print(model3.df_asc_desc_avg)
```

### 4.dfpack2arr

This function is made for data preparation changing df_takers or df_nontakers from table containig five columns (customers, campaign, rev before, rev after, delta). Therefore, no need creating list dataframe.

sample code
```
arr_df_takers, arr_df_nontakers, labels = dfpack2arr('takers_perpack.csv','nontakers_perpack.csv')

```



















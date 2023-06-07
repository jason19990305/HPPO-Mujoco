import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

hppo1 = pd.read_csv('run-HPPO_FetchPush-v120230520-024511-tag-step_success_rate_FetchPush-v1.csv')
hppo2 = pd.read_csv('run-HPPO_FetchPush-v120230520-042628-tag-step_success_rate_FetchPush-v1.csv')
hppo3 = pd.read_csv('run-HPPO_FetchPush-v120230520-060805-tag-step_success_rate_FetchPush-v1.csv')
hppo4 = pd.read_csv('run-HPPO_FetchPush-v120230520-074935-tag-step_success_rate_FetchPush-v1.csv')
hppo5 = pd.read_csv('run-HPPO_FetchPush-v120230520-093059-tag-step_success_rate_FetchPush-v1.csv')
hppo6 = pd.read_csv('run-HPPO_FetchPush-v120230520-111221-tag-step_success_rate_FetchPush-v1.csv')
hppo7 = pd.read_csv('run-HPPO_FetchPush-v120230520-125348-tag-step_success_rate_FetchPush-v1.csv')
hppo8 = pd.read_csv('run-HPPO_FetchPush-v120230520-143520-tag-step_success_rate_FetchPush-v1.csv')
hppo9 = pd.read_csv('run-HPPO_FetchPush-v120230520-161627-tag-step_success_rate_FetchPush-v1.csv')
hppo10 = pd.read_csv('run-HPPO_FetchPush-v120230521-032445-tag-step_success_rate_FetchPush-v1.csv')

x = list()
x.append(hppo1['Step'])

y = list()
y.append(hppo1['Value'])
y.append(hppo2['Value'])
y.append(hppo3['Value'])
y.append(hppo4['Value'])
y.append(hppo5['Value'])
y.append(hppo6['Value'])
y.append(hppo7['Value'])
y.append(hppo8['Value'])
y.append(hppo9['Value'])
y.append(hppo10['Value'])

y = np.array(y)
x = np.array(hppo1['Step'])

y_mean = np.mean(y,axis=0)

y_max = np.max(y,axis=0)
y_min = np.min(y,axis=0)
print(y_max)
print(y_min)

plt.clf()
plt.title('FetchPush-v1')
plt.xlabel('Step')
plt.ylabel('Success rate')
plt.plot(x,y_mean)
plt.legend(['HPPO'])

plt.fill_between(x,y_max,y_min,alpha=0.3)
plt.grid()
plt.show()

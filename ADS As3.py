
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import errors, cluster_tools
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""## Clustering (K-Means)"""

country_codes = ['IDN','GUM','PYF']   # setting country codes to fetch data
ind1=["EN.POP.DNST"]    # setting indicator for Population density
ind1m=["Population density"]   # Indicator name for Population density
ind2=["NY.GDP.MKTP.CD"]    # setting indicator for GDP
ind2m=["GDP"]   # Indicator name for GDP

my_data1  = wb.data.DataFrame(ind1, country_codes, mrv=30).T   # read data for Population density
my_data1=my_data1.fillna(my_data1.median())   # data cleaning
my_data1.head()

plt.figure(figsize=(5,3))
plt.title('Population Density by Year')   # figure title 
plt.plot(my_data1['GUM'],"y")   # plot population density for GUM
plt.plot(my_data1['IDN'],"k")   # plot population density for IDN
plt.plot(my_data1['PYF'],"b")   # plot population density for PYF
plt.xlabel("Year")    # x-label for figure
plt.xticks(rotation=90)   # rotate x-label to 90 degree
plt.ylabel("Population Density")    # x-label for figure
plt.grid()    # apply grid to figure
plt.show()

my_data2  = wb.data.DataFrame(ind2, country_codes, mrv=30).T    # read GDP data
my_data2=my_data2.fillna(my_data2.mean())    # cleaning data
my_data2.head()

plt.figure(figsize=(5,3))
plt.title('GDP of Countries')   # figure title 
plt.plot(my_data2['GUM'],"r",label="GUM")   # plot GDP for GUM
plt.plot(my_data2['IDN'],"k",label="IDN")   # plot GDP for IDN
plt.plot(my_data2['PYF'],"b",label="PYF")   # plot GDP for PYF
plt.xlabel("Year")    # x-label for figure
plt.xticks(rotation=90)   # rotate x-label to 90 degree
plt.ylabel("GDP")    # x-label for figure
plt.legend()   # adding legend in chart
plt.grid()    # apply grid to figure
plt.show()

def heatmap(dt):   # fucntion to call map_corr method for correlation visulaization
    cluster_tools.map_corr(dt)
data_all=[my_data1,my_data2]    # get two data into a list
heatmap(data_all[0])   # call 'heatmap' fucntion for Population density data
heatmap(data_all[1])   # call 'heatmap' fucntion for GDP data

def scaling_data(dt):   # function to call 'scaler' method to normalize data
    res=cluster_tools.scaler(dt)    # normalizwe data
    return res[0], res[1], res[2]
dtscaled=[]
mnv=[]
mxv=[]
for d in range(len(data_all)):
    out=scaling_data(data_all[d])    # get normnalized data, min and max values
    dtscaled.append(out[0])    # storing all normalized data
    mnv.append(out[1])
    mxv.append(out[2])
print(dtscaled[0].head(),"\n")
print(dtscaled[1].head())

def backscldata(dt, mn, mx):    # function top call 'backscale' method tor get back the actuial data from normalized data
    arr=cluster_tools.backscale(dfar,mnv[d],mxv[d])   # call 'backscale' method
    return arr
data_array=[]
for d in range(len(dtscaled)):
    dfar=np.array(dtscaled[d])    # get backscaled data
    arrdt=backscldata(dfar, mnv[d], mxv[d])
    data_array.append(arrdt)
print(data_array[0],"\n")
print(data_array[1])

"""## Clustering"""

def kmclus_elbow(dt,rn):   # function to get inertia values for enblow chart
    vals_el = []
    for i in range(1, rn):   # create loop over clsuter range 1-15(supplied)
        clus_model = KMeans(n_clusters=i, init='k-means++', max_iter=500,  random_state=12)  # apply k-means
        clus_model.fit(dt)   # fit model
        vals_el.append(clus_model.inertia_)   # store inertia
    return vals_el, rn

all_els, elrn=kmclus_elbow(data_array[0],15)
plt.figure(figsize=(5,3))   # figure title 
plt.title('Finding Optimum Cluster')
plt.plot(range(1, elrn), all_els,"m-")   # plot inertia values
plt.plot(range(1, elrn), all_els,"Dr")   # plot inertia values
plt.xlabel('Number of clusters')    # x-label for figure
plt.ylabel('Inertia')    # x-label for figure
plt.grid()   # apply grid to figure
plt.show()

kmfin = KMeans(n_clusters=4, init='k-means++', max_iter=500, n_init=12, random_state=10)   # apply k-means with final cluster value
pred_y = kmfin.fit(data_array[0])   # train model
pred_y

print("Cluster Centers: \n",kmfin.cluster_centers_)

df=pd.DataFrame(data_array[0],columns=my_data1.columns)
plt.figure(figsize=(5,3))   # figure title 
plt.title('Cluster Visualization')
sns.scatterplot(data=df, x="GUM", y="IDN", hue=kmfin.labels_,palette="cividis")   # clsuter visulization
plt.scatter(kmfin.cluster_centers_[:,0], kmfin.cluster_centers_[:,1], marker="d", c="g", s=80, label="centroids")
plt.grid()   # apply grid to figure
plt.legend()   # set best position for legend
plt.show()

"""## Curve Fitting"""

from scipy.optimize import curve_fit

def func(x, a, b, c):   # fucntion to calculating curve fit 
    return a * np.exp(-b * x) + c

y = func(my_data2.values[:,1], 1.6, 0.4, 0.5)
rng = np.random.default_rng()   # apply range
y_noise = 0.55 * rng.normal(size=my_data2.values[:,1].size)   # detercting predictioon error
ydata = y + y_noise    # get the preduicted values
v1, pcov = curve_fit(func, my_data2.values[:,1], ydata)    # apply curve fitting
plt.figure(figsize=(5,3))
plt.title('Results of Curve Fiting')   # figure title 
plt.plot(my_data2.values[:,1], ydata, 'bo', label='data')
plt.plot(my_data2.values[:,1], func(my_data2.values[:,0], *v1), 'g--',label='Best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(v1))
plt.xlabel('x')   # x-label for figure
plt.ylabel('y')    # x-label for figure
plt.legend()   # adding legen in chart
plt.show()


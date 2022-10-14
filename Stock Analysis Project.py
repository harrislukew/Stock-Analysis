#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning: Trade&Ahead
# 
# **Marks: 60**

# ## Problem Statement

# ### Context
# 
# The stock market has consistently proven to be a good place to invest in and save for the future. There are a lot of compelling reasons to invest in stocks. It can help in fighting inflation, create wealth, and also provides some tax benefits. Good steady returns on investments over a long period of time can also grow a lot more than seems possible. Also, thanks to the power of compound interest, the earlier one starts investing, the larger the corpus one can have for retirement. Overall, investing in stocks can help meet life's financial aspirations.
# 
# It is important to maintain a diversified portfolio when investing in stocks in order to maximise earnings under any market condition. Having a diversified portfolio tends to yield higher returns and face lower risk by tempering potential losses when the market is down. It is often easy to get lost in a sea of financial metrics to analyze while determining the worth of a stock, and doing the same for a multitude of stocks to identify the right picks for an individual can be a tedious task. By doing a cluster analysis, one can identify stocks that exhibit similar characteristics and ones which exhibit minimum correlation. This will help investors better analyze stocks across different market segments and help protect against risks that could make the portfolio vulnerable to losses.
# 
# 
# ### Objective
# 
# Trade&Ahead is a financial consultancy firm who provide their customers with personalized investment strategies. They have hired you as a Data Scientist and provided you with data comprising stock price and some financial indicators for a few companies listed under the New York Stock Exchange. They have assigned you the tasks of analyzing the data, grouping the stocks based on the attributes provided, and sharing insights about the characteristics of each group.
# 
# ### Data Dictionary
# 
# - Ticker Symbol: An abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market
# - Company: Name of the company
# - GICS Sector: The specific economic sector assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - GICS Sub Industry: The specific sub-industry group assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - Current Price: Current stock price in dollars
# - Price Change: Percentage change in the stock price in 13 weeks
# - Volatility: Standard deviation of the stock price over the past 13 weeks
# - ROE: A measure of financial performance calculated by dividing net income by shareholders' equity (shareholders' equity is equal to a company's assets minus its debt)
# - Cash Ratio: The ratio of a  company's total reserves of cash and cash equivalents to its total current liabilities
# - Net Cash Flow: The difference between a company's cash inflows and outflows (in dollars)
# - Net Income: Revenues minus expenses, interest, and taxes (in dollars)
# - Earnings Per Share: Company's net profit divided by the number of common shares it has outstanding (in dollars)
# - Estimated Shares Outstanding: Company's stock currently held by all its shareholders
# - P/E Ratio: Ratio of the company's current stock price to the earnings per share 
# - P/B Ratio: Ratio of the company's stock price per share by its book value per share (book value of a company is the net difference between that company's total assets and total liabilities)

# ## Importing necessary libraries

# In[2]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist, pdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# to suppress warnings
import warnings
warnings.filterwarnings("ignore")


# ## Loading the dataset

# In[3]:


## import the data
data = pd.read_csv(r'C:\Users\harri\OneDrive\Documents\Great Learning Projects\Project 7\stock_data.csv')


# ## Overview of the Dataset

# The initial steps to get an overview of any dataset is to: 
# - observe the first few rows of the dataset, to check whether the dataset has been loaded properly or not
# - get information about the number of rows and columns in the dataset
# - find out the data types of the columns to ensure that data is stored in the preferred format and the value of each property is as expected.
# - check the statistical summary of the dataset to get an overview of the numerical columns of the data

# In[ ]:





# ### Checking the shape of the dataset

# In[4]:


# checking shape of the data
data.shape## get the shape of data


# ### Displaying few rows of the dataset

# In[9]:


# let's view a sample of the data
data.sample(n=10, random_state=1)


# ### Checking the data types of the columns for the dataset

# In[10]:


# checking the column names and datatypes
data.info()


# ### Creating a copy of original data

# In[11]:


# copying the data to another variable to avoid any changes to original data
df = data.copy()


# ### Checking for duplicates and missing values

# In[12]:


# checking for duplicate values
df.duplicated().sum() ## get total number of duplicate values


# In[13]:


# checking for missing values in the data
df.isnull().sum() ## check the missing values in the data


# ### Statistical summary of the dataset

# **Let's check the statistical summary of the data.**

# In[14]:


df.describe(include='all').T


# ## Exploratory Data Analysis

# ### Univariate analysis

# In[15]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(df, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=df, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        df[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        df[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# **`Current Price`**

# In[16]:


histogram_boxplot(df, 'Current Price')


# **`Price Change`**

# In[18]:


histogram_boxplot(df, 'Price Change')  ## create histogram_boxplot for 'Price Change'


# **`Volatility`**

# In[19]:


histogram_boxplot(df, 'Volatility')  ## create histogram_boxplot for 'Volatility'


# **`ROE`**

# In[20]:


histogram_boxplot(df, 'ROE')  ## create histogram_boxplot for 'ROE'


# **`Cash Ratio`**

# In[21]:


histogram_boxplot(df, 'Cash Ratio')  ## create histogram_boxplot for 'Cash Ratio'


# **`Net Cash Flow`**

# In[22]:


histogram_boxplot(df, 'Net Cash Flow')  ## create histogram_boxplot for 'Net Cash Flow'


# **`Net Income`**

# In[23]:


histogram_boxplot(df, 'Net Income')  ## create histogram_boxplot for 'Net Income'


# **`Earnings Per Share`**

# In[24]:


histogram_boxplot(df, 'Earnings Per Share')  ## create histogram_boxplot for 'Earnings Per Share'


# **`Estimated Shares Outstanding`**

# In[25]:


histogram_boxplot(df, 'Estimated Shares Outstanding')  ## create histogram_boxplot for 'Estimated Shares Outstanding'


# **`P/E Ratio`**

# In[26]:


histogram_boxplot(df, 'P/E Ratio')  ## create histogram_boxplot for 'P/E Ratio'


# **`P/B Ratio`**

# In[27]:


histogram_boxplot(df, 'P/B Ratio')  ## create histogram_boxplot for 'P/B Ratio'


# In[28]:


# function to create labeled barplots


def labeled_barplot(df, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(df[feature])  # length of the column
    count = df[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=df,
        x=feature,
        palette="Paired",
        order=df[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# **`GICS Sector`**

# In[29]:


labeled_barplot(df, 'GICS Sector', perc=True)


# **`GICS Sub Industry`**

# In[30]:


labeled_barplot(df, 'GICS Sub Industry', perc = True)  ## create a labelled barplot for 'GICS Sub Industry'


# ### Bivariate Analysis

# In[31]:


# correlation check
plt.figure(figsize=(15, 7))
sns.heatmap(
    df.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# **Let's check the stocks of which economic sector have seen the maximum price increase on average.**

# In[33]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='Price Change', ci=False)  ## choose the right variables
plt.xticks(rotation=90)
plt.show()


# **Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. Let's see how the average cash ratio varies across economic sectors.**

# In[32]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='Cash Ratio', y='GICS Sector', ci=False)  ## choose the right variables
plt.xticks(rotation=90)
plt.show()


# **P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. Let's see how the P/E ratio varies, on average, across economic sectors.**

# In[34]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='P/E Ratio', ci=False)  ## choose the right variables
plt.xticks(rotation=90)
plt.show()


# **Volatility accounts for the fluctuation in the stock price. A stock with high volatility will witness sharper price changes, making it a riskier investment. Let's see how volatility varies, on average, across economic sectors.**

# In[35]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='Volatility', ci=False)  ## choose the right variables
plt.xticks(rotation=90)
plt.show()


# ## Data Preprocessing

# ### Outlier Check
# 
# - Let's plot the boxplots of all numerical columns to check for outliers.

# In[36]:


plt.figure(figsize=(15, 12))

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(numeric_columns):
    plt.subplot(3, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### Scaling
# 
# - Let's scale the data before we proceed with clustering.

# In[38]:


num_col = df.select_dtypes(include=np.number).columns.tolist()


# In[39]:


num_col


# In[40]:


# scaling the data before clustering
scaler = StandardScaler()
subset = df[num_col].copy()  ## scale the data
subset_scaled = scaler.fit_transform(subset)


# In[41]:


# creating a dataframe of the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)


# ## K-means Clustering

# ### Checking Elbow Plot

# In[42]:


k_means_df = subset_scaled_df.copy()


# In[43]:


clusters = range(1, 15)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(subset_scaled_df)
    prediction = model.predict(k_means_df)
    distortion = (
        sum(np.min(cdist(k_means_df, model.cluster_centers_, "euclidean"), axis=1))
        / k_means_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)
plt.show()


# In[66]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1, 15), timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure


# ### Let's check the silhouette scores

# In[45]:


sil_score = []
cluster_list = range(2, 15)
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    preds = clusterer.fit_predict((subset_scaled_df))
    score = silhouette_score(k_means_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, the silhouette score is {})".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.show()


# In[52]:


get_ipython().system('pip install yellowbrick')
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


# In[50]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2, 15), metric="silhouette", timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure


# In[71]:


# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(6, random_state=1))  ## visualize the silhouette scores for certain number of clusters
visualizer.fit(k_means_df)
visualizer.show()


# ### Creating Final Model

# In[93]:


# final K-means model
kmeans = KMeans(n_clusters=6, random_state=1)  ## choose the number of clusters
kmeans.fit(k_means_df)


# In[94]:


# creating a copy of the original data
df1 = df.copy()

# adding kmeans cluster labels to the original and scaled dataframes
k_means_df["KM_segments"] = kmeans.labels_
df1["KM_segments"] = kmeans.labels_


# ### Cluster Profiling

# In[95]:


km_cluster_profile = df1.groupby("KM_segments").mean()  ## to groupby the cluster labels


# In[96]:


km_cluster_profile["count_in_each_segment"] = (
    df1.groupby("KM_segments")["Security"].count().values  ## groupby the cluster labels
)


# In[97]:


km_cluster_profile.style.highlight_max(color="blue", axis=0)


# In[98]:


## print the companies in each cluster
for cl in df1["KM_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df1[df1["KM_segments"] == cl]["Security"].unique())
    print()


# In[99]:


df1.groupby(["KM_segments", "GICS Sector"])['Security'].count()


# In[100]:


plt.figure(figsize=(20, 20))
plt.suptitle("Boxplot of numerical variables for each cluster")

# selecting numerical columns
num_col = df.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df1, x="KM_segments", y=variable)

plt.tight_layout(pad=2.0)


# ### Insights

# - 
# 

# ## Hierarchical Clustering

# ### Computing Cophenetic Correlation

# In[63]:


hc_df = subset_scaled_df.copy()


# In[107]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"] ## add distance metrics

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"] ## add linkages

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(hc_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(hc_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm
            
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print('*'*100)
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# **Let's explore different linkage methods with Euclidean distance only.**

# In[108]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"] ## add linkages

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(hc_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(hc_df))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm
        
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print('*'*100)
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)


# **Let's view the dendrograms for the different linkage methods with Euclidean distance.**

# ### Checking Dendrograms

# In[109]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"] ## add linkages

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]
compare = []

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(hc_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(hc_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )

    compare.append([method, coph_corr])


# In[110]:


# create and print a dataframe to compare cophenetic correlations for different linkage methods
df_cc = pd.DataFrame(compare, columns=compare_cols)
df_cc = df_cc.sort_values(by="Cophenetic Coefficient")
df_cc


# ### Creating model using sklearn

# In[111]:


HCmodel = AgglomerativeClustering(n_clusters=7, affinity="euclidean", linkage="ward")  ## define the hierarchical clustering model
HCmodel.fit(hc_df)


# In[112]:


# creating a copy of the original data
df2 = df.copy()

# adding hierarchical cluster labels to the original and scaled dataframes
hc_df["HC_segments"] = HCmodel.labels_
df2["HC_segments"] = HCmodel.labels_


# ### Cluster Profiling

# In[113]:


hc_cluster_profile = df2.groupby("HC_segments").mean()  ## groupby the cluster labels


# In[114]:


hc_cluster_profile["count_in_each_segment"] = (
    df2.groupby("HC_segments")["Security"].count().values  ## groupby the cluster labels
)


# In[119]:


hc_cluster_profile.style.highlight_max(color="blue", axis=0)


# In[116]:


## print the companies in each cluster
for cl in df2["HC_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df2[df2["HC_segments"] == cl]["Security"].unique())
    print()


# In[117]:


df2.groupby(["HC_segments", "GICS Sector"])['Security'].count()


# In[118]:


plt.figure(figsize=(20, 20))
plt.suptitle("Boxplot of numerical variables for each cluster")

for i, variable in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df2, x="HC_segments", y=variable)

plt.tight_layout(pad=2.0)


# ## K-means vs Hierarchical Clustering

# You compare several things, like:
# - Which clustering technique took less time for execution?
# - Which clustering technique gave you more distinct clusters, or are they the same?
# - How many observations are there in the similar clusters of both algorithms?
# - How many clusters are obtained as the appropriate number of clusters from both algorithms?
# 
# You can also mention any differences or similarities you obtained in the cluster profiles from both the clustering techniques.

# In[ ]:





# ## Actionable Insights and Recommendations

# - 
# 

# ___

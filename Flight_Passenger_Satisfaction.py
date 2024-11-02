#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Importing Data

# In[ ]:


import pandas as pd
train = pd.read_csv("/content/train (2).csv")
test = pd.read_csv("/content/test (2).csv")


# In[ ]:


train.shape


# In[ ]:


train.head(10)


# In[ ]:


train = train.drop('Unnamed: 0', axis=1)
train = train.drop('id', axis=1)


# In[ ]:


train.info()


# Repeating the same steps for test data set as well...

# In[ ]:


test.shape


# In[ ]:


test.head(10)


# In[ ]:


test = test.drop('Unnamed: 0', axis=1)
test = test.drop('id', axis=1)


# In[ ]:


test.info()


# In[ ]:


train.columns = [c.replace(' ', '_') for c in train.columns]


# In[ ]:


test.columns = [c.replace(' ', '_') for c in test.columns]


# In[ ]:


train['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)


# In[ ]:


test['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)


# # Checking for Imbalance

# In[ ]:


# balanced or imbalanced
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,5))
train.satisfaction.value_counts(normalize = True).plot(kind='bar', color= ['darkorange','steelblue'], alpha = 0.9, rot=0)
plt.title('Satisfaction Indicator (0) and (1) in the Dataset')
plt.show()


# The above plot shows a distribution of around 55%:45% between neutral/dissatisfied passengers and satisfied passengers respectively. So the data is quite balanced and it does not require any special treatment/resampling.
# 
# # Handling of Missing Data

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head()


# In[ ]:


# Replacing missing values with mean
train['Arrival_Delay_in_Minutes'] = train['Arrival_Delay_in_Minutes'].fillna(train['Arrival_Delay_in_Minutes'].mean())


# In[ ]:


test['Arrival_Delay_in_Minutes'] = test['Arrival_Delay_in_Minutes'].fillna(test['Arrival_Delay_in_Minutes'].mean())


# In[ ]:


train.select_dtypes(include=['object']).columns


# In[ ]:


# Replacing missing values with mode
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
train['Customer_Type'] = train['Customer_Type'].fillna(train['Customer_Type'].mode()[0])
train['Type_of_Travel'] = train['Type_of_Travel'].fillna(train['Type_of_Travel'].mode()[0])
train['Class'] = train['Class'].fillna(train['Class'].mode()[0])


# In[ ]:


test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
test['Customer_Type'] = test['Customer_Type'].fillna(test['Customer_Type'].mode()[0])
test['Type_of_Travel'] = test['Type_of_Travel'].fillna(test['Type_of_Travel'].mode()[0])
test['Class'] = test['Class'].fillna(test['Class'].mode()[0])


# # Exploratory Data Analysis

# In[ ]:


import seaborn as sns
with sns.axes_style(style='ticks'):
    g = sns.catplot("satisfaction", col="Gender", col_wrap=2, data=train, kind="count", height=2.5, aspect=1.0)  
    g = sns.catplot("satisfaction", col="Customer_Type", col_wrap=2, data=train, kind="count", height=2.5, aspect=1.0)


# **Gender:** 
# It is observed that gender-wise distribution of dissatisfied and satisfied customers are quite same. For both male and female passengers, no. of dissatisfied customers are on the higher side compared to no. of satisfied customers.
# 
# **Customer Type:**
# Loyal passengers are very high in number. Even among loyal passengers, the ratio of satisfied and dissatidfied ones are almost 49:51. 

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot("Age", data=train, aspect=3.0, kind='count', hue='satisfaction', order=range(5, 80))
    g.set_ylabels('Age vs Passenger Satisfaction')


# **Age:**
# From age 7-to-38 and from age 61-to-79, quotient of dissatisfied passengers is very high compared to satisfied passengers. On the contrary, in age range 39-60, quotient of satisfied passengers is higher compared to dissatisfied passengers.

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot(x="Flight_Distance", y="Type_of_Travel", hue="satisfaction", col="Class", data=train, kind="bar", height=4.5, aspect=.8)


# **Type of Travel, Class, Flight Distance:**
# For business travel in business class category, the number of satisfied passengers are quite on the higher side for longer flight distance. For other combinations, there is almost equal distribution of satisfied and dissatisfied passengers.
# 

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot(x="Departure/Arrival_time_convenient", y="Online_boarding", hue="satisfaction", col="Class", data=train, kind="bar", height=4.5, aspect=.8)


# **Online Boarding, Departure/Arrival Time Convenience grouped by Class:**
# For Eco Plus class, very inconvenient Departure/Arrival time (Departure/Arrival_time_convenient = 0) has really high no. of dissatisfied passengers, even when online boarding is done very well. For other combinations, no. of satisfied passengers are on the higher side compared to no. of dissatisfied passengers. 

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot(x="Class", y="Departure_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=train, kind="bar", height=4.5, aspect=.8)
    g = sns.catplot(x="Class", y="Arrival_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=train, kind="bar", height=4.5, aspect=.8)


# **Departure Delay, Arrival Delay grouped by Type of Travel:**
# For personal travel (specially Eco Plus and Eco), the no. of dissatisfied passengers are really high when arrival delay in minutes is high. Now, this is quite obvious. By minute comparison, all combinations have higher no. of dissatisfied passengers compared to no. of satisfied passengers.

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot(x="Gate_location", y="Baggage_handling", hue="satisfaction", col="Class", data=train, kind="box", height=4.5, aspect=.8)


# **Baggage Handling, Gate Location grouped by Class:**
# For business class, it is observed that all gate locations have higher no. of dissatisfied passengers when baggage handling is not done perfectly well (rating <= 4). For Eco Plus, when the gate location is 1 and for Eco, when the gate location is 2, even when the baggages are handled in a mediocre way (rating in range 2.0 - 4.0), passengers remained dissatisfied.

# In[ ]:


with sns.axes_style('white'):
    g = sns.catplot(x="Inflight_wifi_service", y="Inflight_entertainment", hue="satisfaction", col="Class", data=train, kind="box", height=4.5, aspect=.8)


# **Inflight Entertainment, Inflight wi-fi Service grouped by Class:**
# It is interesting to find that Eco Plus passengers are mostly satisfied without in-flight wi-fi service (rating 0) and medium level of in-flight entertainment (rating 2 - 4). For Business class passengers, only highest level of in-flight entertainment (rating 5) can make them satisfied. For Eco passengers, high level of in-flight entertainment (rating 3 - 5) and very high wi-fi service availability (rating 5) can make them satisfied.

# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("satisfaction", col="Ease_of_Online_booking", col_wrap=6, data=train, kind="count", height=2.5, aspect=.9)


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("satisfaction", col="Seat_comfort", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("satisfaction", col="Cleanliness", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("satisfaction", col="Food_and_drink", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)


# **Ease of Online Booking, Seat Comfort, Cleanliness, Food and Drink:**
# For all of these features, maximum no. of satisfied passengers belong to the category of 4 and 5 rating givers. Below rating 4, passengers are mostly dissatisfied.

# In[ ]:


import matplotlib.pyplot as plt 
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

table1 = pd.crosstab(train['satisfaction'], train['Checkin_service'])
sns.heatmap(table1, cmap='Oranges', ax = axarr[0][0])
table2 = pd.crosstab(train['satisfaction'], train['Inflight_service'])
sns.heatmap(table2, cmap='Blues', ax = axarr[0][1])
table3 = pd.crosstab(train['satisfaction'], train['On-board_service'])
sns.heatmap(table3, cmap='pink', ax = axarr[1][0])
table4 = pd.crosstab(train['satisfaction'], train['Leg_room_service'])
sns.heatmap(table4, cmap='bone', ax = axarr[1][1])


# **Checkin Service, Inflight Service, On-board Service, Leg-room Service:**
# For checkin service, 0-2 rating givers are predominantly dissatisfied. For other three services, only 4 and 5 rating givers belong to satisfied passengers category.  

# # Label Encoding of Categorical Variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in train.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    train[col] = lencoders[col].fit_transform(train[col])


# In[ ]:


lencoders_t = {}
for col in test.select_dtypes(include=['object']).columns:
    lencoders_t[col] = LabelEncoder()
    test[col] = lencoders_t[col].fit_transform(test[col])


# # Outliers Detection and Removal

# In[ ]:


Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


train = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
train.shape


# # Correlation among Features

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(150, 1, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})


# "Ease_of_Online_booking" is highly correlated with "Inflight_wifi_service". Also "Inflight_service" is highly correlated with "Baggage_handling". But no pair is having corr. coefficient exactly equal to 1. So there is no perfect multicollinearity. Hence we are not discarding any variable. 
# 
# # Top 10 Feature Selection through Chi-Square

# In[ ]:


from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(train)
#modified_data = pd.DataFrame(r_scaler.transform(train), index=train['id'], columns=train.columns)
modified_data = pd.DataFrame(r_scaler.transform(train), columns=train.columns)
modified_data.head()


# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='satisfaction']
y = modified_data[['satisfaction']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


# These are top 10 features impacting on passenger satisfaction. We will check feature importance with other methods as well.
# # Feature Importance using Wrapper Method

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf

X = train.drop('satisfaction', axis=1)
y = train['satisfaction']
selector = SelectFromModel(rf(n_estimators=100, random_state=0))
selector.fit(X, y)
support = selector.get_support()
features = X.loc[:,support].columns.tolist()
print(features)
print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)


# So only these six features are inherently important in contributing towards passenger satisfaction.
# 

# # Building Models

# In[ ]:


features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
            'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
            'Inflight_service', 'Baggage_handling']
target = ['satisfaction']

# Split into test and train
X_train = train[features]
y_train = train[target].to_numpy()
X_test = test[features]
y_test = test[target].to_numpy()

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


import time
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_confusion_matrix, plot_roc_curve
from matplotlib import pyplot as plt 
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train.ravel(), verbose=0)
    else:
        model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.pink, normalize = 'all')
    plot_roc_curve(model, X_test, y_test)                     
    
    return model, accuracy, roc_auc, time_taken


# **Model-1: Logistic Regression penalized with Elastic Net (L1 penalty = 50%, L2 penalty = 50%)**

# In[ ]:


from sklearn.linear_model import LogisticRegression

params_lr = {'penalty': 'elasticnet', 'l1_ratio':0.5, 'solver': 'saga'}

model_lr = LogisticRegression(**params_lr)
model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, X_train, y_train, X_test, y_test)


# Since Logistic Regression is a white-box model (explainable), we can dive deeper into it to get more insight. 

# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())


# We can see, among 12 features, except 6th feature (Inflight_entertainment), rest 11 features have p-value < 0.05. So these are really important features impacting highly towards the target variable. Also, a pseudo R-square value **(McFadden's Pseudo R-Squared Value)** of 0.55 represents an excellent fit. 
# 
# **Model-2: Naive Bayes Classifier**

# In[ ]:


from sklearn.naive_bayes import GaussianNB

params_nb = {}

model_nb = GaussianNB(**params_nb)
model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_model(model_nb, X_train, y_train, X_test, y_test)


# **Model-3: K-Nearest Neighbor Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

params_kn = {'n_neighbors':10, 'algorithm': 'kd_tree', 'n_jobs':4}

model_kn = KNeighborsClassifier(**params_kn)
model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_kn, X_train, y_train, X_test, y_test)


# Model-4: Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
params_dt = {'max_depth': 12,    
             'max_features': "sqrt"}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)


# Since Decision Tree is a white-box (explainable) model, we can deep-dive into its visualization to get more valuable insight below. From tree-visualization, we can extract rules which are contributing towards passenger-satisfaction.

# In[ ]:


import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

features_n = ['Type_of_Travel', 'Inflight_wifi_service', 'Online_boarding', 'Seat_comfort']
X_train_n = scaler.fit_transform(train[features_n])
data = export_graphviz(DecisionTreeClassifier(max_depth=3).fit(X_train_n, y_train), out_file=None, 
                       feature_names = features_n,
                       class_names = ['Dissatisfied (0)', 'Satisfied (1)'], 
                       filled = True, rounded = True, special_characters = True)
# we have intentionally kept max_depth short here to accommodate the entire visual-tree, best result comes with max_depth = 12
# we have taken only really important features here to accommodate the entire tree picture
graph = graphviz.Source(data)
graph


# From above tree visualization, it can be easily spotted that rule "Type_of_Travel <=0.227 and Seat_comfort <= -0.089 and Online_boarding <= 0.045" (all normalized values) contributes towards passenger satisfaction indicator= 1. Like that, many other rules can be extracted easily by going through the nodes.
# 
# 

# **Model-5: Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)


# Well, we see that Random Forest has performed very well on both Accuracy and area under ROC curve. So, we are now interested to see **how many decision trees are minimally required make the Accuarcy consistent** (recalling the fact that Random Forest is actually a bagged ensemble of decision trees).

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

trees=range(100)
accuracy=np.zeros(100)

for i in range(len(trees)):
    clf = RandomForestClassifier(n_estimators = i+1)
    model1 = clf.fit(X_train, y_train.ravel())
    y_predictions = model1.predict(X_test)
    accuracy[i] = accuracy_score(y_test, y_predictions)

plt.plot(trees,accuracy)


# From above graph, it is evident that **minimum 40 trees** are required to make accuracy fairly consistent (though minimal fluctuation is still there, and we can try the graph after increasing the no. of iterations).
# 

# **Model - 6: Neural Network (Multilayer Perceptron)**

# In[ ]:


from sklearn.neural_network import MLPClassifier

params_nn = {'hidden_layer_sizes': (30,30,30),
             'activation': 'logistic',
             'solver': 'lbfgs',
             'max_iter': 100}

model_nn = MLPClassifier(**params_nn)
model_nn, accuracy_nn, roc_auc_nn, tt_nn = run_model(model_nn, X_train, y_train, X_test, y_test)


# **Model-7: Extreme Gradient Boosting**

# In[ ]:


import xgboost as xgb
params_xgb ={'n_estimators': 500,
            'max_depth': 16}

model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)


# **Model-8: Adaptive Gradient Boosting**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier as adab
params_adab ={'n_estimators': 500,
              'random_state': 12345}

model_adab = adab(**params_adab)
model_adab, accuracy_adab, roc_auc_adab, tt_adab = run_model(model_adab, X_train, y_train, X_test, y_test)


# **Model-9 : SVM**

# In[ ]:


from sklearn.svm import SVC # "Support vector classifier"

clf = SVC()

clf, accuracy_svc, roc_auc_svc, tt_svc = run_model(clf, X_train, y_train, X_test, y_test)


# # Hyperparameter Tuning
# 

# Hyperparameter tuning  is the process of determining the right combination of hyperparameters that maximizes the model performance

# **From the above comparision of models the best model fit to our dataset is Random Forest.** So we perform hyper parameter tuning to random forest to decide the best parameters to train our model to increase its accuracy than before.

# In the grid search method, we create a grid of possible values for hyperparameters. Each iteration tries a combination of hyperparameters in a specific order. It fits the model on each and every combination of hyperparameters possible and records the model performance. Finally, it returns the best model with the best hyperparameters.

# **Random Forest**

# In[ ]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200, 300]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


grid_search.fit(X_train, y_train).best_estimator_


# We got fine-tuned the max_features and n_estimators parameters of the random forest algorithm.

# In[ ]:


grid_search.best_score_


# **Accuracy for Random Forest has increased from 0.89 to 0.96 after hyperparameter tuning**
# 
# This shows that our model has improved and will produce an optimal solution.

# # UNSUPERVISED:

# In[ ]:


airline = pd.read_csv("/content/Airline_Dataset.csv")
airline


# In[ ]:


airline.isnull().sum()


# In[ ]:


airline["Arrival Delay in Minutes"].fillna("0.0", inplace = True)


# In[ ]:


airline.isnull().sum()


# In[ ]:


train.isnull().sum()


# In[ ]:


airline


# **PCA**

# In[ ]:


def delay_transformation(x):
    if x<=10:
        x = "less_than_10min"
    elif x<=40:
        x = "10_to_40min"
    elif x<=120:
        x = "41_to_120min"
    elif x<=240:
        x = "121_240min"
    elif x<=450:
        x = "241_450min"
    else:
        x = "more_than_450min"

    return x


# In[ ]:


def transform_dataset(df, clean=True, transform=True, scaler=True):
    
    if clean==True:

        df.set_index("id", inplace=True)

        df.Gender = df.Gender.apply(lambda x: 1 if x=="Female" else 0)

        df.sort_values(by="id",inplace=True)

        df.rename(columns={"Customer Type":"Loyal"}, inplace=True)
        df["Loyal"] = df["Loyal"].apply(lambda x: 1 if x=='Loyal Customer' else 0)

        df.rename(columns={"satisfaction":"Dissatisfied"}, inplace=True)
        df["Dissatisfied"] = df['Dissatisfied'].apply(lambda x: 1 if x=='neutral or dissatisfied' else 0)

        df.rename(columns={'Type of Travel':"Business Travel"}, inplace=True)
        df["Business Travel"] = df["Business Travel"].apply(lambda x: 1 if x=='Business travel' else 0)
    
    if transform == True:
        df["Flight Distance"] = np.log(df["Flight Distance"]+1)

        df["Departure Delay in Minutes"] = df["Departure Delay in Minutes"].apply(delay_transformation)
        df.rename(columns={"Departure Delay in Minutes":"Departure_Delay"},inplace=True)

        df.drop(columns="Arrival Delay in Minutes", inplace=True)

        df = pd.get_dummies(df, columns=["Class","Departure_Delay"],drop_first=True)
     
    if scaler==True:
        df.loc[:,["Age","Flight Distance"]] = StandardScaler().fit_transform(df.loc[:,["Age","Flight Distance"]])
        
        df.loc[:,"Inflight wifi service":"Cleanliness"] = StandardScaler().fit_transform(df.loc[:,"Inflight wifi service":"Cleanliness"])
    
    return df


# In[ ]:


airline_train_transf = transform_dataset(pd.read_csv("/content/Airline_Dataset.csv"))


# In[ ]:


# Creating PCA model:
from sklearn.decomposition import PCA
def pca_transformation(df, n):
    pca_model = PCA(n_components=n)
    
    df = pca_model.fit_transform(df)
    
    columns_list = []
    for i in range(1,n+1):
        columns_list.append("PC"+str(i))
    
    df = pd.DataFrame(df, columns=columns_list)
    
    return df


# In[ ]:


airline_train_pca = pca_transformation(airline_train_transf, 3)


# In[ ]:


airline_train_pca


# In[ ]:


X_airline_train = airline_train_transf.drop(columns='Dissatisfied').copy()

print(X_airline_train.shape)

Y_airline_train = airline_train_transf['Dissatisfied'].copy()
print(Y_airline_train.shape)


# In[ ]:


X_airline_train = transform_dataset(pd.read_csv("/content/Airline_Dataset.csv").copy()).drop("Dissatisfied",axis=1)

# Applying PCA without target column
X_airline_train = pca_transformation(X_airline_train, 6)

Y_airline_train = transform_dataset(pd.read_csv("/content/Airline_Dataset.csv"))["Dissatisfied"]


# In[ ]:


X_airline_train.shape


# In[ ]:


airline_test_transf = transform_dataset(pd.read_csv("/content/Airline_Dataset.csv"))


# In[ ]:


X_airline_test = airline_test_transf.drop(columns='Dissatisfied').copy()
# Applying PCA without target column
X_airline_test = pca_transformation(X_airline_test, 6)

print(X_airline_test.shape)

Y_airline_test = airline_test_transf['Dissatisfied'].copy()
print(Y_airline_test.shape)


# In[ ]:


Y_airline_test.value_counts()/Y_airline_test.shape[0]


# **K Means**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from yellowbrick.cluster import KElbowVisualizer

# Visualizing the elbow choosing the best number of clusters
visualizer=KElbowVisualizer(KMeans(n_clusters = 9), k=(2,10))#, metric='silhouette')
visualizer.fit(airline_test_transf)
visualizer.poof()


# In[ ]:


kmeans_model = KMeans(n_clusters=4)

# Fitting the model
kmeans_model.fit(X_airline_train)


# In[ ]:


kmeans_model = KMeans(n_clusters=4)

# Fitting the model
kmeans_model.fit(X_airline_train)
KMeans_y_preds = kmeans_model.predict(X_airline_test)
KMeans_preds = pd.DataFrame({"Predicted":KMeans_y_preds,"Actual":Y_airline_test})
KMeans_preds


# In[ ]:


accuracy = accuracy_score(Y_airline_test, KMeans_y_preds)
roc_auc = roc_auc_score(Y_airline_test, KMeans_y_preds) 
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc))


# In[ ]:


print("Number of clusters defined:", len(set(kmeans_model.labels_)))


# In[ ]:


airline_train_cluster = transform_dataset(pd.read_csv("/content/Airline_Dataset.csv"),transform=False,scaler=False)

airline_train_cluster["Cluster_KMeans"] = kmeans_model.labels_

airline_train_cluster


# In[ ]:


airline_train_cluster.groupby("Cluster_KMeans").mean()


# In[ ]:


plt.figure(figsize=(16,16))
sns.jointplot(data=airline_train_cluster, x="Age",y="Flight Distance",hue="Cluster_KMeans", kind='kde')


# In[ ]:


cluster_airline_train = pd.merge(X_airline_train, Y_airline_train, on=Y_airline_train.index)

cluster_airline_train.set_index("key_0",inplace=True)

cluster_airline_train["Cluster_KMeans"] = kmeans_model.labels_

cluster_airline_train


# In[ ]:


cluster_airline_train.groupby('Cluster_KMeans').mean()


# In[ ]:


sns.scatterplot(data=cluster_airline_train, x="PC1",y="PC2", hue="Cluster_KMeans")


# # DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


dbscan_model = DBSCAN(eps=1.1)

# Model fitting
dbscan_model.fit(X_airline_train)

# Storing the result in a column of our new DataFrame
airline_train_cluster["Cluster_DBSCAN"] = dbscan_model.labels_


# In[ ]:


print("Number of clusters:", len(set(dbscan_model.labels_)))


# In[ ]:


# Looking at the different clusters created:
airline_train_cluster.groupby("Cluster_DBSCAN").mean()


# In[ ]:


airline_train_cluster.groupby("Cluster_DBSCAN").count()


# In[ ]:


train.describe()


# In[ ]:


import plotly.express as px
fig = px.scatter_3d(cluster_airline_train,x='PC1',y='PC2',z='PC3',color='Dissatisfied')
fig.show()


# Conclusion: the DBSCAN doesn't perform well in this case, either creating a huge number of clusters or creating not enough clusters. Since this model is too slow to run and it provokes very different results for a slight modification in eps, it is hard to adopt and optimise this model.

# **Agglomerative Clustering**
# 
# Hierarchical Clustering
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df2=train[:5000]
# df = df.sample(frac=0.50)


# In[ ]:


y1 = df2['satisfaction']
x1 = df2.drop('satisfaction',axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[ ]:


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test= train_test_split(x1, y1, test_size=0.2, random_state=42)


# In[ ]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms Representation")  
dend = shc.dendrogram(shc.linkage(df2, method='ward'))


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df2)


# In[ ]:


cluster.fit(x1_train)
cluster_y_preds = cluster.fit_predict(x1_test)


# In[ ]:


agcl_preds = pd.DataFrame({"Predicted":cluster_y_preds,"Actual":y1_test})
agcl_preds


# In[ ]:


agcl_TP = len(agcl_preds[(agcl_preds["Predicted"]==agcl_preds["Actual"])&(agcl_preds["Predicted"]==1)])
agcl_FP = len(agcl_preds[(agcl_preds["Predicted"]!=agcl_preds["Actual"])&(agcl_preds["Predicted"]==1)])
agcl_FN = len(agcl_preds[(agcl_preds["Predicted"]!=agcl_preds["Actual"])&(agcl_preds["Predicted"]==0)])
agcl_TN = len(agcl_preds[(agcl_preds["Predicted"]==agcl_preds["Actual"])&(agcl_preds["Predicted"]==0)])
print(agcl_TP,agcl_FP,agcl_FN,agcl_TN)
print("Rightly Classified: ",(agcl_TP+agcl_TN),"/",(agcl_TP+agcl_FP+agcl_FN+agcl_TN))
print("Wrongly Classified: ",(agcl_FP+agcl_FN),"/",(agcl_TP+agcl_FP+agcl_FN+agcl_TN))

agcl_Accuracy = (agcl_TP+agcl_TN)/(agcl_TP+agcl_FP+agcl_FN+agcl_TN)
agcl_Precision = (agcl_TP)/(agcl_TP+agcl_FP)
agcl_Recall = (agcl_TP)/(agcl_TP+agcl_FN)
agcl_Specificity = (agcl_TN)/(agcl_TN+agcl_FP)
agcl_F1 = (2*agcl_Precision*agcl_Recall)/(agcl_Precision+agcl_Recall)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(y1_test, cluster_y_preds)
print("Accuracy: {:.2f}%".format(accuracy * 100))




# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y1_test, cluster_y_preds)
plt.figure()
sns.heatmap(cf, annot=True)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')


# In[ ]:


from sklearn.metrics import silhouette_score

silhouette_scores = []
silhouette_scores.append(
        silhouette_score(df2, cluster.fit_predict(df2)))
silhouette_scores.append(
        silhouette_score(df2, ac3.fit_predict(df2)))
silhouette_scores.append(
        silhouette_score(df2, ac2.fit_predict(df2)))


# In[ ]:


plt.bar([2,3,4], silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
cluster1 = AgglomerativeClustering(n_clusters=2, affinity='manhattan', linkage='average')  
cluster1.fit_predict(df2)


# In[ ]:


plt.figure(figsize=(10, 7))  
plt.scatter(df2['Age'], df2['Flight_Distance'], c=cluster1.labels_,cmap ='rainbow') 


# In[ ]:


ac33 = AgglomerativeClustering(n_clusters = 3, affinity='manhattan', linkage='average')
  
plt.figure(figsize =(6, 6))
plt.scatter(df2['Age'], df2['Flight_Distance'],
           c = ac33.fit_predict(df2), cmap ='rainbow')
plt.show()


# In[ ]:


ac22 = AgglomerativeClustering(n_clusters = 4, affinity='manhattan', linkage='average')
  
plt.figure(figsize =(6, 6))
plt.scatter(df2['Age'], df2['Flight_Distance'], c = ac22.fit_predict(df2), cmap ='rainbow')
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score

silhouette_scores1 = []
silhouette_scores1.append(
        silhouette_score(df2, cluster.fit_predict(df2)))
silhouette_scores1.append(
        silhouette_score(df2, ac33.fit_predict(df2)))
silhouette_scores1.append(
        silhouette_score(df2, ac22.fit_predict(df2)))


# In[ ]:


silhouette_scores


# In[ ]:


silhouette_scores1


# In[ ]:


plt.bar([2,3,4], silhouette_scores1)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


# In[ ]:


comparison = pd.DataFrame({"ML Classification Algo":["Agglomerative Clustering"],
 "Rightly_Classified":[(agcl_TP+agcl_TN)/(agcl_TP+agcl_FP+agcl_FN+agcl_TN)],
 "Wrongly_Classified":[(agcl_FP+agcl_FN)/(agcl_TP+agcl_FP+agcl_FN+agcl_TN)],
 "Accuracy":[agcl_Accuracy],
 "Precision":[agcl_Precision],
 "Recall":[agcl_Recall],
 "Specificity":[agcl_Specificity],
 "F1-Score":[agcl_F1]})

comparison.sort_values(by="Accuracy",ascending=False).style.background_gradient(cmap='rainbow')


# # Model Comparison:
# We will compare the performace of various models by their respective ROC_AUC score and total time taken for execution.

# In[ ]:


roc_auc_scores = [roc_auc_lr, roc_auc_nb, roc_auc_kn, roc_auc_dt, roc_auc_nn, roc_auc_rf, roc_auc_xgb, roc_auc_adab]
tt = [tt_lr, tt_nb, tt_kn, tt_dt, tt_nn, tt_rf, tt_xgb, tt_adab]

model_data = {'Model': ['Logistic Regression','Naive Bayes','K-NN','Decision Tree','Neural Network','Random Forest','XGBoost','AdaBoost'],
              'ROC_AUC': roc_auc_scores,
              'Time taken': tt}
data = pd.DataFrame(model_data)

fig, ax1 = plt.subplots(figsize=(14,8))
ax1.set_title('Model Comparison: Area under ROC Curve and Time taken for execution by Various Models', fontsize=13)
color = 'tab:blue'
ax1.set_xlabel('Model', fontsize=13)
ax1.set_ylabel('Time taken', fontsize=13, color=color)
ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette='Blues_r')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('ROC_AUC', fontsize=13, color=color)
ax2 = sns.lineplot(x='Model', y='ROC_AUC', data = data, sort=False, color=color)
ax2.tick_params(axis='y', color=color)


# # Conclusion
# We observe, Random Forest have performed well on producing high ROC_AUC score (90%). Also **Random Forest** has taken lesser amount of time . So, we will stick to Random Forest as the best model. After performing hyperparameter tuning the accuracy score has increased to 96%
# 


# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_mldata

mnist =fetch_mldata('mnist original',data_home="./data")PCA for A Small Subset


# In[2]:


import numpy as np

np.random.seed(123)

#Shuffle data
permutation = np.random.permutation(mnist.data.shape[0])
X = mnist.data[permutation]
y = mnist.target[permutation]

# Split data
N_train = 60000

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]


# # Visualization Using PCA

# In[3]:


from sklearn.decomposition import PCA


# In[4]:


my_pca=PCA(n_components=2)


# In[6]:


my_pca.fit(X)


# In[7]:


X_hat = my_pca.transform(X)
X_hat


# In[11]:


import matplotlib.pyplot as plt

sc = plt.scatter(X_hat[:, 0], X_hat[:, 1],c=y)
plt.colorbar(sc)
plt.show()


# # PCA for A Small Subset

# In[12]:


from sklearn.decomposition import PCA

pca_for_0_1 = PCA(n_components=2)
pca_for_0_1.fit(X[y < 2])

X_hat = pca_for_0_1.transform(X[y < 2])


# In[13]:


import matplotlib.pyplot as plt

sc = plt.scatter(X_hat[:, 0], X_hat[:, 1], c=y[y < 2])
plt.colorbar(sc)
plt.show()


# # Bonus: LDA (Extra Material)

# In[14]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)


# In[15]:


X_hat = lda.transform(X)
X_hat

sc = plt.scatter(X_hat[:, 0], X_hat[:, 1], c=y)
plt.colorbar(sc)
plt.show()


# # Clustering

# In[17]:


from sklearn.cluster import KMeans

my_kmeans = KMeans(n_clusters=10)
my_kmeans.fit(X)


# In[18]:


clusters=my_kmeans.predict(X)


# In[22]:


X_hat=my_pca.transform(X)

sc = plt.scatter(X_hat[:, 0], X_hat[:, 1], c=clusters)
plt.colorbar(sc)
plt.show()


# In[21]:


fig, array = plt.subplots(2, 5)
for i, index in enumerate(np.random.choice(
                            np.where(clusters == 0)[0], 10)):
    image = np.reshape(X[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
plt.show()


# In[24]:


fig, array = plt.subplots(2, 5)
for i, index in enumerate(np.random.choice(
                            np.where(clusters == 1)[0], 10)):
    image = np.reshape(X[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
plt.show()


# In[25]:


fig, array = plt.subplots(2, 5)
for i, index in enumerate(np.random.choice(
                            np.where(clusters == 2)[0], 10)):
    image = np.reshape(X[index], (28, 28))
    array[int(i/5), i%5].imshow(image, cmap='gray')
plt.show()


# In[26]:


from sklearn import metrics

print('score:',metrics.adjusted_rand_score(y,clusters))


# # What If We Reduce Dimentions First?
# 

# In[27]:


pca_50 = PCA(n_components=50)
pca_50.fit(X)

X_50 = pca_50.transform(X)


# In[ ]:


my_kmeans=KMeans(n_clusters=20)
my_kmeans.fit(X_50)

clusters_2 = my_kmeans.predict(X_50)


# In[ ]:


X_hat = my_pca.transform(X)

sc = plt.scatter(X_hat[:, 0], X_hat[:, 1], c=clusters_2)
plt.colorbar(sc)
plt.show()


# In[ ]:


from sklearn import metrics

print("score: ", metrics.adjusted_rand_score(y, clusters_2))


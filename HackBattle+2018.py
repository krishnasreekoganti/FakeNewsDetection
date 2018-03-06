
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("fake_or_real_news.csv")
df.shape
df.shapedf = df.set_index("Unnamed: 0")
y = df.label 
df.drop("label", axis=1)


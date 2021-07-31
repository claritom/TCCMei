
# coding: utf-8

# TCC Classificação de municípios com base em indicadores do MEI
# Importação e Preparação dos dados
# 28/06/2021

# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_mei = pd.read_excel('inadimplencia.xlsx')
df_mei.info()


# In[4]:


df_mei.head(20)


# In[9]:


df_mei.drop(columns='nm_municipio', inplace=True)


# In[10]:


# IBGE - população
df_populacao = pd.read_excel('populacao.xlsx')
df_populacao.info()


# In[11]:


df_populacao['cod_mun'] = df_populacao['cod_uf'].apply('{:0>2}'.format) + df_populacao['cod_municipio'].apply('{:0>5}'.format)
df_populacao['cod_mun'] = df_populacao['cod_mun'].astype(int)
df_populacao.head(20)


# In[12]:


df_populacao.info()


# In[13]:


# IBGE - estimativa PIB
df_pib = pd.read_excel('pib.xlsx')
df_pib.info()


# In[14]:


# Junção MEI x População
df_completo = pd.merge(df_mei, df_populacao[['cod_mun','uf','qtd_populacao_estimada']], on='cod_mun')
#df_completo[['DAS_pagos','qtd_populacao_estimada']].head()
df_completo.head()


# In[15]:


# Junção PIB
df_completo = pd.merge(df_completo, df_pib, on='cod_mun')
#df_completo[['DAS_pagos','qtd_populacao_estimada']].head()


# In[16]:


df_completo.head(20)


# In[17]:


df_completo.info()


# In[18]:


# Exportação do dataset completo
df_completo.to_csv('completo.csv', mode='w',          
               index=False , header=True , encoding='utf-8', sep='|',line_terminator='\r\n',float_format='%.2f')


# In[19]:


# Verifica valores nulos
#df_completo.isna().all(1)
#for k in range (1,xx)
df_nulos = df_completo[df_completo['vlr_adic_bruto_agropecuaria'].isnull()]
df_nulos


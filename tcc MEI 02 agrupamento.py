
# coding: utf-8

# TCC Classificação de municípios com base em indicadores do MEI
# Análise e Agrupamento
# 28/06/2021

# In[121]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


sns.__version__


# In[122]:


# Importação do dataset completo após tratamento e merge
df_completo = pd.read_csv('completo.csv', header=0, delimiter='|')

df_completo.info()


# In[123]:


# Verifica valores nulos
#df_completo.isna().all(1)
#for k in range (1,xx)
df_nulos = df_completo[df_completo['vlr_adic_bruto_agropecuaria'].isnull()]
df_nulos


# In[124]:


# Trata inadimplencia negativa (altera para 0)
df_completo['inadimplencia'].loc[df_completo['inadimplencia']<0] = 0   #.count()
#df_inadimplencia_neg


# In[125]:


df_completo.loc[df_completo['inadimplencia']<0]

df_completo['inadimplencia'].describe()


# In[126]:


df_completo['porte'] = pd.cut(df_completo['qtd_populacao_estimada'], [0, 25000, 50000, 100000, 1000000, 15000000], labels=["P1 Pequeno","P2 Pequeno","P3 Médio", "P4 Grande", "P5 Metrópole"])


# 4. Análise e Exploração dos Dados

# In[127]:


df_completo[['DAS_pagos','qtd_optantes','inadimplencia','vlr_adic_bruto_total','impostos_liquidos_sobre_produtos','pib','pib_per_capita']].describe()


# In[128]:


df_completo['qtd_populacao_estimada'].describe().apply("{0:f}".format)


# In[129]:


relat = df_completo.groupby(by=['porte'])
relat['porte'].count()


# In[130]:


relat[['porte','DAS_pagos','qtd_optantes','inadimplencia','qtd_populacao_estimada']].mean()


# In[131]:


df_completo[['porte','vlr_adic_bruto_administracao','vlr_adic_bruto_total','impostos_liquidos_sobre_produtos','pib','pib_per_capita']].groupby(by=['porte']).mean()


# In[132]:


sns.set(style='whitegrid') # v 0.9.0

# Distribuição inadimplência
sns.distplot(df_completo['inadimplencia'])


# In[7]:


sns.countplot(x='porte', data=df_completo)


# In[135]:


plt.scatter(df_completo['qtd_populacao_estimada'], df_completo['qtd_optantes'], s=30) 
plt.title('Optantes x População')
plt.xlabel('População')
plt.ylabel('Optantes')

#plt.legend()
plt.show()


# In[8]:


# Sem tratamento de inadimplencia negativa
sns.boxplot(x='porte', y='inadimplencia', data=df_completo)


# In[18]:


# Após tratamento de inadimplencia negativa (alterado para 0)
sns.boxplot(x='porte', y='inadimplencia', data=df_completo)


# In[9]:


sns.barplot(x='porte', y='inadimplencia', data=df_completo)


# In[211]:


df_uf = df_completo[['uf','qtd_optantes','inadimplencia', 'vlr_adic_bruto_total']].groupby('uf')
relat = df_uf.mean().sort_values(by='inadimplencia', ascending=False)
relat['qtd_municipios'] = df_uf['uf'].count() 
relat


# In[162]:


# Valores não numéricos em vlr_adic_bruto_agropecuaria
df_completo[~df_completo['vlr_adic_bruto_agropecuaria'].apply(lambda x: np.isreal(x))]


# In[137]:


# Matriz de Correlação de Pearson
fig, ax = plt.subplots(figsize=(15,10))

df_pearson = sns.heatmap(df_completo[['DAS_pagos','qtd_optantes','adimplencia','inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_agropecuaria','vlr_adic_bruto_industria','vlr_adic_bruto_servicos','vlr_adic_bruto_administracao','vlr_adic_bruto_total','impostos_liquidos_sobre_produtos','pib','pib_per_capita']].corr(method='pearson'), annot=True) # 


# In[138]:


df_completo.info()


# In[139]:


# Normalização dos dados com MinMaxScaler

df_prep = df_completo[['inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_agropecuaria','vlr_adic_bruto_industria','vlr_adic_bruto_servicos','vlr_adic_bruto_administracao','vlr_adic_bruto_total']]
scaler = MinMaxScaler()

print(scaler.fit(df_prep))
print(scaler.data_max_)

arr_prep_sc = scaler.transform(df_prep)

#print(df_prep)
#arr_prep_sc


# In[140]:


# Identifica n_clusters ideal
kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": None,
 }

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(arr_prep_sc)
    sse.append(kmeans.inertia_)
    
print(sse)


# In[141]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.title('Método Elbow')
plt.xlabel("Número de Clusters")
plt.ylabel("SSE")
plt.show()


# In[153]:


# Aplicação do algoritmo k-means com o número ideal de clusters (3) calculado no passo anterior e com 5
kmeans = KMeans(n_clusters = 5)
kmeans.fit(arr_prep_sc)


# In[157]:


# O método FIT retorna os rótulos dos dados, a predição. Atribuir esse retorno a uma variável e utilizá-la em outra partes do código
df_clusters = df_completo
df_clusters['cluster'] = kmeans.fit_predict(arr_prep_sc)


# In[144]:


df_clusters.info()


# In[161]:


# Exportação do dataset agrupado
df_clusters.to_csv('clusters.csv', mode='w',          
               index=False , header=True , encoding='utf-8', sep='|',line_terminator='\r\n',float_format='%.2f')


# In[162]:


df_clusters[['cluster','inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_total']].groupby(by=['cluster']).describe()


# In[170]:


df_clusters[['cluster','porte','cod_mun']].groupby(by=['cluster','porte']).count()


# In[171]:


kmeans.cluster_centers_


# In[172]:


kmeans.labels_

#df_prep_sc['cluster'] = 


# In[173]:


# Atributos: 'qtd_optantes','inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_total'
plt.scatter(arr_prep_sc[:,1], arr_prep_sc[:,0], c=kmeans.labels_, s=50)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0],  c='red', label='Centroides', s = 50)
plt.title('n_clusters = 5')
plt.xlabel('População')
plt.ylabel('Inadimplência')
#plt.yscale('linear')
plt.legend()

plt.show()


# In[174]:


plt.scatter(arr_prep_sc[:,3], arr_prep_sc[:,1], c = kmeans.labels_, s=100) 
plt.scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 1],  c = 'red',label = 'Centroides', s=100)
plt.title('Inadimplência')
plt.xlabel('Valor adicionado bruto total')
plt.ylabel('Inadimplência')
#plt.yscale('linear')
plt.legend()

plt.show()


# In[ ]:





# In[103]:


sns.set(style='whitegrid') # v 0.9.0

# Distribuição inadimplência
sns.distplot(df_completo['qtd_populacao_estimada'].loc[df_clusters['cluster'] == 3])


# In[83]:


# Cluster 0 - maior inadimplência)
df_cluster_max = df_completo[df_completo['cluster'] == 3].sort_values(by=['inadimplencia'], ascending=True)

df_cluster_max.describe()


# In[209]:


df_porte = df_cluster_max[['porte','inadimplencia','qtd_optantes']].groupby(by='porte')
relat = df_porte.mean() 
relat['qtd'] = df_porte['porte'].count() 
relat


# In[88]:


df_cluster_max[df_cluster_max['porte'] == 'P5 Metrópole']


# In[206]:


df_uf = df_cluster_max[['uf','inadimplencia','qtd_optantes']].groupby(by='uf')
relat = df_uf.mean().sort_values(by='inadimplencia', ascending=False)
relat['qtd'] = df_uf['uf'].count() 
relat


# In[181]:


# Cluster 1 - inadimplência média - municípios menores
df_cluster_med = df_clusters[df_clusters['cluster'] == 1].sort_values(by=['inadimplencia'], ascending=True)

df_cluster_med[['inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_total']].describe()


# In[207]:


df_porte = df_cluster_med[['porte','inadimplencia','qtd_optantes']].groupby(by='porte')
relat = df_porte.mean() 
relat['qtd'] = df_porte['porte'].count() 
relat


# In[184]:


sns.boxplot(x='porte', y='inadimplencia', data=df_cluster_med, linewidth=2)


# In[185]:


# Cluster 2 - menor inadimplência
df_cluster_min = df_clusters[df_completo['cluster'] == 2].sort_values(by=['inadimplencia'], ascending=True)

df_cluster_min.describe()
#df_cluster_min[df_cluster_min['inadimplencia'] > 0].head(10)


# In[208]:


df_porte = df_cluster_min[['porte','inadimplencia','qtd_optantes']].groupby(by='porte')
relat = df_porte.mean() 
relat['qtd'] = df_porte['porte'].count() 
relat


# In[187]:


df_grafico = df_completo[df_completo['cluster'] == 2]

plt.scatter(df_grafico['qtd_populacao_estimada'], df_grafico['inadimplencia'], s=50) 
#plt.scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 1],  c = 'red',label = 'Centroides', s=100)
plt.title('Cluster 2 - menor inadimplência')
plt.xlabel('População')
plt.ylabel('Inadimplência')

#plt.legend()

plt.show()


# In[189]:


df_cluster_min[df_cluster_min['qtd_populacao_estimada'] > 25000]


# In[190]:


# Cluster 3 - média alta - municípios maiores
df_cluster_medA = df_completo[df_clusters['cluster'] == 3].sort_values(by=['inadimplencia'], ascending=True)

df_cluster_medA[['inadimplencia','qtd_populacao_estimada','vlr_adic_bruto_total']].describe()


# In[191]:


df_cluster_medA[['porte','inadimplencia','qtd_populacao_estimada']].groupby(by='porte').describe()


# In[193]:


sns.boxplot(x='porte', y='inadimplencia', data=df_cluster_medA, linewidth=2)


# In[194]:


df_metropole = df_cluster_medA[df_cluster_medA['porte'] == 'P5 Metrópole']
df_metropole[df_cluster_medA['inadimplencia'] > 0.5]


# In[195]:


# Cluster 4 - inadimplência baixa
df_cluster_bxa = df_clusters[df_clusters['cluster'] == 4].sort_values(by=['inadimplencia'], ascending=True)

df_cluster_bxa.describe()


# In[196]:


df_porte = df_cluster_bxa[['porte','inadimplencia','qtd_populacao_estimada']].groupby(by='porte')
relat = df_porte.mean() 
relat['qtd'] = df_porte['porte'].count() 
relat


# In[197]:


df_cluster_bxa[df_cluster_bxa['porte'] == 'P4 Grande']


# In[199]:


# Metrópoles
df_clusters[['cluster','cod_mun','nm_municipio','uf','DAS_pagos','qtd_optantes','inadimplencia','qtd_populacao_estimada']].loc[df_completo['porte'] == 'P5 Metrópole']


# In[201]:


df_clusters[['uf','cluster']].loc[df_completo['cluster'] == 0].groupby(by=['uf']).count()


# In[202]:


df_clusters[['uf','qtd_optantes','inadimplencia']].loc[df_clusters['cluster'] == 0].groupby('uf').mean().sort_values(by='inadimplencia', ascending=False)


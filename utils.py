import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from sklearn.manifold import TSNE
import umap

def toumap(fts):
    umap_reduc = umap.UMAP(n_components=2, random_state=42)
    #X = np.array(X)
    X = np.array(fts)
    return umap_reduc.fit_transform(X)

def plot_simple(X, labels, title):
    cmap = plt.cm.get_cmap('tab20', 35)

    # Obter as cores dos mapas de cores tab20 e tab10
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab10 = plt.cm.rainbow(np.linspace(0, 1, 15))

    # Combinar as cores para criar um novo conjunto de cores
    combined_colors = colors_tab10 #np.vstack((colors_tab20, colors_tab10))

    #for label, color in zip(unique_labels, colors):
    #    plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], cmap=cmap, s=10,  edgecolors='k', linewidths=0.1)
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Dark2", edgecolor='black', s=15, linewidth=0.1)
    #scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(combined_colors), edgecolor='black', s=10, linewidth=0.1)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('t-SNE Visualization of Data with 30 Classes')
    plt.colorbar(scatter)

    plt.title(title)
    plt.show()

def totsne(fts):
    tsne = TSNE(n_components=2, random_state=42)
    #X = np.array(X)
    X = np.array(fts)
    return tsne.fit_transform(X)

def plotsmart(X, labels, title = "", nolabels = False):
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if nolabels:
            plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], c=[color], s=5)
        else: 
            plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], c=[color], s=5, label=f'Cluster {label}')


    plt.legend()
    plt.title(title)
    plt.show()



# ==============================================================================================================
# ============================================ RANKS FUNCS =====================================================
# ==============================================================================================================


from scipy.spatial.distance import pdist, squareform

def getmap(ranking, showrank = False):
  ctdrelev = 0
  rates = []
  ratesrev = []
  idclass = ranking[1]
  ranking = ranking[0]

  for i, r in enumerate(ranking):
      ctdrelev += 1 if r[1] == idclass else 0
      rates.append(ctdrelev / (i+1))
      if r[1] == idclass:
        ratesrev.append(ctdrelev / (i+1))

  if showrank: print(ranking)
  #mapgeral = sum(ratesrev) / len(ratesrev)
  mapgeral = 0 if len(ratesrev)<1 else sum(ratesrev) / len(ratesrev)
  return mapgeral

def getranks(_feats, l=0):
    if l<1: l = len(_feats)

    # Converter a lista de vetores para numpy array
    vetores_array = np.array(_feats)

    # Calcular a matriz de distâncias entre todos os vetores
    distancias_condensadas = pdist(vetores_array, metric='euclidean')
    matriz_distancias = squareform(distancias_condensadas)

    def neareast(vetor_index):
        # Obter as distâncias do vetor_index para todos os outros
        distancias = matriz_distancias[vetor_index]        
        # Obter os índices dos vetores ordenados pela distância, ignorando o próprio vetor
        indices_ordenados = np.argsort(distancias)[:l]        
        return indices_ordenados[1:], distancias[indices_ordenados[1:]]  # Ignorando o próprio vetor

    def getrk(_idx):
        vetores_proximos, distancias_proximos = neareast(_idx)
        return [ [idx, distancias_proximos[i]] for i, idx in enumerate(vetores_proximos)]

    return [ getrk(i) for i in range(len(_feats))  ]

def getrank(feats, fanchor, idx):
    if type(feats)!=torch.Tensor:
        feats = torch.stack(feats)
    # Remova a característica da âncora se idx for válido
    if idx > -1:
        feats = torch.cat((feats[:idx], feats[idx+1:]), dim=0)

    # Expanda fanchor para que tenha a mesma forma que feats
    fanchor_expanded = fanchor.unsqueeze(0).expand_as(feats)

    # Calcule todas as distâncias de uma vez
    distances = F.pairwise_distance(fanchor_expanded, feats)

    # Construa a lista de pares [distância, índice]
    ranking = [[dist.item(), i if i < idx else i + 1] for i, dist in enumerate(distances)]

    # Ordene pelo primeiro elemento (distância)
    ranking.sort(key=lambda x: x[0])

    # Retorne os índices e as distâncias ordenadas
    return [[r[1], r[0]] for r in ranking]

def getallranks(fs, rk_length = 0, intervalshow = 0):    
    if rk_length<1: rk_length = len(fs)
    
    rks = []
    for i, f in enumerate(fs):
        rks.append(getrank(fs, f, i)[:rk_length])
        if intervalshow > 0 and (i % intervalshow)==0:
            print("Ranks generated:", i)
    return rks

def getallmap(tsfs, lbs):
    ranks = [ getrank(tsfs, f, i) for i, f in enumerate(tsfs)  ]
    rks = [ [[ [r[0], lbs[r[0]], r[1]] for r in rk], lbs[j]] for j, rk in enumerate(ranks)]
    maps = [getmap(r) for r in rks]
    return np.mean(maps)

def calcmap(rks, lbs):
    rks = [ [[ [r[0], lbs[r[0]], r[1]] for r in rk], lbs[j]] for j, rk in enumerate(rks)]
    maps = [getmap(r) for r in rks]
    return np.mean(maps)


# ==============================================================================================================
# ========================================== CLUSTERS FUNCS ====================================================
# ==============================================================================================================


import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def bestk_silhouette_score(X, nfactor = 1.1, rangeinit= 3, rangeend = 100, showscores = False):
    silhouette_scores = []
    K_range = range(rangeinit, rangeend)

    for k in K_range:        
        agg = AgglomerativeClustering(n_clusters=k)
        agg.fit(X)
        labels = agg.labels_
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        # Inertia apenas com kmeans
        # inertia.append(km.inertia_) # Para Elbow Method: escolher o ponto onde o ganho de melhoria começa a diminuir (explorar mais pra frente)
        print("K score", k, ":", score)

    if showscores:
        # Plot do Silhouette Score
        plt.plot(K_range, silhouette_scores, 'bo-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score para diferentes k')
        plt.show()

    bestscore = max(silhouette_scores)
    return round([k for k, score in zip(K_range, silhouette_scores) if score==bestscore][0] * nfactor)


def getclusters_ws(fs, clusterlbs):
    clusterids = list(set(clusterlbs))
    centroids_feats = [ torch.mean(torch.stack([d[0] for d in zip(fs, clusterlbs) if d[1]==cl]), dim=0) for cl in clusterids]

    c_rks = getranks(centroids_feats)

    # Normalization
    alldists = [rk for rks in [[r[1] for r in c] for c in c_rks] for rk in rks]
    vmax = max(alldists)
    vmin = min(alldists)
    for ck in c_rks:
        for k in ck:
            k[1] = (k[1]-vmin)/(vmax-vmin)

    clusterdic = {}
    for c, ck in enumerate(c_rks):
        ckdic = {}
        ckdic[c] = 0
        for k in ck:        
            ckdic[k[0]]= k[1]       
        clusterdic[c] = ckdic
    return clusterdic
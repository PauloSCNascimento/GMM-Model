import random
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import plt
random.seed(42) # define the seed (important to reproduce the results)
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from numpy import unique
from numpy import where
import pandas as pd




############################################################
############################################################
# Classe GMM    
class GMM:
     # Classe GMM para dados 1 dimensão    

    ##############################################
    ##############################################
    def __init__(self, num_clusters, max_iterations,colors):
        #Inicializar dados e max_iterations
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.colors = colors
    ##############################################

    ##############################################
    ##############################################        
    def Initialization(self, X):
        #Inicialize os parâmetros e execute as etapas E e M armazenando o valor de log da verossimilhança após cada iteração"
        self.pi = np.zeros(self.num_clusters)
        self.mu = np.zeros((self.num_clusters, len(X[0])))
        self.cov = np.zeros((self.num_clusters, len(X[0]), len(X[0])))
        
        M_Kmeans = KMeans(n_clusters=self.num_clusters)
        # fit do modelo
        M_Kmeans.fit(X)
        # atribuir um cluster para cada exemplo
        yhat = M_Kmeans.predict(X)
        #  recuperar clusters únicos
        clusters = unique(yhat)
        # criar gráfico de dispersão para amostras de cada cluster
        for r in clusters:
            # obter índices de linha para amostras com este cluster
            row_ix = where(yhat == r)
            self.mu[r]=np.sum(X[row_ix,:][0], axis=0)/len(X[row_ix,:][0])
            self.cov[r,:,:]=np.cov((X[row_ix,:][0]).T)
            self.pi[r]=len(X[row_ix,:][0])/len(X)
        self.pi=self.pi/np.sum(self.pi)
        
        # self.pi = [1/3,1/3,1/3]
        # self.mu = [[0,2.5],[5,17],[10,10]]
    ##############################################

    ##############################################
    ##############################################
    def run(self, X):
        #Execução dos passos EM
       
        self.MCov_Corr = 1e-6*np.identity(len(X[0])) #is used for numerical stability i.e. to check singularity issues in covariance matrix 

        ################################
        ################################
        # Plot os dados e o modelo inicial
        x,y = np.meshgrid(np.sort(X[:,0]), np.sort(X[:,1]))
        self.XY = np.array([x.flatten(), y.flatten()]).T

        plt.figure(figsize=(10,10))
        plt.scatter(X[:, 0], X[:, 1])
        plt.title("Initial State")        
        
        for m, c in zip(self.mu, self.cov):
            c += self.MCov_Corr
            multi_normal = multivariate_normal(mean=m, cov=c)
            plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)), colors = 'black', alpha = 0.3)
            plt.scatter(m[0], m[1], c='grey', zorder=10, s=100)
        
        plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
        plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
        plt.xlabel('$X_{1}$')
        plt.ylabel('$X_{2}$')
        plt.show()
        ################################
        
        self.log_likelihoods = []

        for iters in range(self.max_iterations):
            ################################
            ################################
            # E-Step
            self.ric = np.zeros((len(X), len(self.mu)))#Inicializa a matriz de score (Nº amostras x Nº clusters)

            for pic, muc, covc, r in zip(self.pi, self.mu, self.cov, range(len(self.ric[0]))):
                covc += self.MCov_Corr#Adiciona valores baixos pra evitar singularidade na matriz de covariância
                mn = multivariate_normal(mean=muc, cov=covc)
                self.ric[:, r] = pic*mn.pdf(X)#Calcula a probabilidade de cada amostra pertencer a cada cluster r

            for r in range(len(self.ric)):
                self.ric[r, :] = self.ric[r, :] / np.sum(self.ric[r, :])#Normaliza as probabilidades, de forma a soma das probabilidades da amostra em cada cluster seja 1

            ################################
            
            ################################
            ################################
            # M-Step
            self.mc = np.sum(self.ric, axis=0)#Calcula a soma das probabilidades de todos os elementos pra cada cluster
            self.mc[where(self.mc == 0)]=1e-6            
            self.pi = self.mc/np.sum(self.mc)#Novo peso de cada gaussiana é dado pela probabilidade dos elementos pertecerem a cada cluster
            self.mu = np.dot(self.ric.T, X) / self.mc.reshape(self.num_clusters,1)#Novas média de cluster é dada pela média ponderada pelas probabilidades das amostras nos clusters

            self.cov = []

            for r in range(len(self.pi)):
                #Nova matrix de covariância é dada distância de Mahalanobis poderanda pela probabilidade de cada elemento pertencer o cluster r
                #1/mc *Sum[ric * (X-mu)*(X-mu)']
                covc = 1/self.mc[r] * (np.dot( ( self.ric[:, r].reshape(len(X), 1) * (X-self.mu[r]) ).T , (X - self.mu[r]) ) + self.MCov_Corr)
                self.cov.append(covc)

            self.cov = np.asarray(self.cov)
            ################################
            
            ################################
            ################################
            #Calcula a verossimilhança aproximada, ponderando cada gaussiana pelo peso pi
            likelihood_sum = np.sum([self.pi[r]*multivariate_normal(self.mu[r], self.cov[r] + self.MCov_Corr).pdf(X) for r in range(len(self.pi))])
            self.log_likelihoods.append(np.log(likelihood_sum))#Aplica o log na verossimilhança aproximada
            ################################
            
            ################################
            ################################            
            # Plot os dados e o modelo a cada iteração            
            plt.figure(figsize=(10,10))
            plt.scatter(X[:, 0], X[:, 1])
            plt.title("Iteração " + str(iters))

            for m, c in zip(self.mu, self.cov):
                c += self.MCov_Corr
                multi_normal = multivariate_normal(mean=m, cov=c)
                plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)), colors = 'black', alpha = 0.3)
                plt.scatter(m[0], m[1], c='grey', zorder=10, s=100)
            
            plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
            plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
            plt.xlabel('$X_{1}$')
            plt.ylabel('$X_{2}$')
            plt.show()
            ################################

        ################################      
        ################################
        #Plot da evolução do log da verossimilhança
        plt.figure(figsize=(10,10))
        plt.plot(range(0, iters+1, 1), self.log_likelihoods)
        plt.title('Valores do log da verossimilhança por iteração')
        plt.xlabel('Nº iterações')
        plt.ylabel('Val. log ver.')
        plt.show()
        ################################
    ##############################################
    
    ##############################################
    ##############################################
    def predict(self, Y):
        #Previsão de cluster para novas amostras na matriz Y
        ################################
        ################################
        predictions = []
        for pic, m, c in zip(self.pi, self.mu, self.cov):
            prob = pic*multivariate_normal(mean=m, cov=c).pdf(Y)
            predictions.append([prob])

        predictions = np.asarray(predictions).reshape(self.num_clusters,len(Y))
        predictions = np.argmax(predictions.T, axis=1)
        ################################
        
        ################################
        ################################
        #Plot os dados previstos
        plt.figure(figsize=(10,10))
        plt.scatter(X[:, 0], X[:, 1], c='c')
        plt.scatter(Y[:, 0], Y[:, 1], marker='*', c='k', s=150, label = 'New Data')
        plt.xlabel('$X_{1}$')
        plt.ylabel('$X_{2}$')
        plt.title("Previsões")
            
        colors=self.colors
    
        for m, c, col, i in zip(self.mu, self.cov, colors, range(len(colors))):
    #         c += MCov_Corr
            multi_normal = multivariate_normal(mean=m, cov=c)
            plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)), colors = 'black', alpha = 0.3)
            plt.scatter(m[0], m[1], marker='o', c=col, zorder=10, s=150, label = 'Centroid ' + str(i+1))

        for i in range(len(Y)):
            plt.scatter(Y[i, 0], Y[i, 1], marker='*', c=colors[predictions[i]], s=150)

        plt.xlabel('$X_{1}$')
        plt.ylabel('$X_{2}$')
        plt.legend()
        plt.show()
        ################################
        return predictions
    ##############################################
############################################################    
  
    


############################################################
############################################################
# Definição dos dados entrada para clusterização 
NClusters=3
Namostra=900   
colors = ['r', 'b', 'g']


# data = pd.read_csv('data/data/Dados das Usinas.csv', header=(0))
# data = data.dropna(axis='rows') #remove NaN
# data = data.to_numpy()
# nrow,ncol = data.shape
# X = data

mean = [[0, 5],[5,15],[9,15]]
cov = [[[1, -0.5], [-0.5, 1]],[[1, 0.8], [0.8, 1]],[[1, 0], [0, 1]]]
xc1 = np.random.multivariate_normal(mean[0], cov[0],int(Namostra/NClusters))
xc2 = np.random.multivariate_normal(mean[1], cov[1],int(Namostra/NClusters))
xc3 = np.random.multivariate_normal(mean[2], cov[2],int(Namostra/NClusters))
X=np.asarray([xc1,xc2,xc3]).reshape(Namostra,2)

plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("Dados originais")
plt.show()        
############################################################

    
############################################################
############################################################
# Execução do modelo GMM    
gmm = GMM(num_clusters=NClusters, max_iterations=10,colors=colors)
gmm.Initialization(X)
gmm.run(X)
############################################################


############################################################
############################################################
# Predição do GMM implementado
Y=X
yhat =gmm.predict(Y)

# encontrar os clusters
clusters = unique(yhat)
# criar gráfico de dispersão para amostras de cada cluster
for cluster in clusters:
    # obter índices de linha para amostras com este cluster
    row_ix = where(yhat == cluster)
    # criar dispersão dessas amostras
    plt.scatter(Y[row_ix, 0], Y[row_ix, 1],c=colors[cluster], label = 'Centroid ' + str(cluster+1))
# mostrar os gráficos
plt.legend()
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("GMM implementado")
plt.show() 
############################################################


############################################################
############################################################
# Predição do GMM Pacote Python
from sklearn.mixture import GaussianMixture
GMM = GaussianMixture(n_components=NClusters)
GMM.fit(X)
Y=X
yhat = GMM.predict(Y)
# encontrar os clusters
clusters = unique(yhat)
# criar gráfico de dispersão para amostras de cada cluster
for cluster in clusters:
    # obter índices de linha para amostras com este cluster
    row_ix = where(yhat == cluster)
    # criar dispersão dessas amostras
    plt.scatter(Y[row_ix, 0], Y[row_ix, 1],c=colors[cluster], label = 'Centroid ' + str(cluster+1))
# mostrar os gráficos
plt.legend()
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("GMM Pacote Python")
plt.show() 
############################################################

############################################################
############################################################
# Predição do KMeans Pacote Python
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=NClusters)
Model=KM.fit(X)
Y=X
yhat = KM.predict(Y)
# encontrar os clusters
clusters = unique(yhat)
# criar gráfico de dispersão para amostras de cada cluster
for cluster in clusters:
    # obter índices de linha para amostras com este cluster
    row_ix = where(yhat == cluster)
    # criar dispersão dessas amostras
    plt.scatter(Y[row_ix, 0], Y[row_ix, 1],c=colors[cluster], label = 'Centroid ' + str(cluster+1))
# mostrar os gráficos
plt.legend()
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("KMeans Pacote Python")
plt.show() 
############################################################



############################################################
############################################################
# Superfície de separação do GMM  Pacote Python
x = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
y = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
x, y = np.meshgrid(x, y)
Y = np.array([x.flatten(), y.flatten()]).T
yhat = GMM.predict(Y)
# encontrar os clusters
clusters = unique(yhat)
# criar gráfico de dispersão para amostras de cada cluster
for cluster in clusters:
    # obter índices de linha para amostras com este cluster
    row_ix = where(yhat == cluster)
    # criar dispersão dessas amostras
    plt.scatter(Y[row_ix, 0], Y[row_ix, 1],c=colors[cluster], label = 'Centroid ' + str(cluster+1))
# mostrar os gráficos
plt.legend()
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("GMM Pacote Python")
plt.show() 
############################################################

############################################################
############################################################
# Superfície de separação do KMeans Pacote Python
x = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
y = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
x, y = np.meshgrid(x, y)
Y = np.array([x.flatten(), y.flatten()]).T
yhat = KM.predict(Y)
# encontrar os clusters
clusters = unique(yhat)
# criar gráfico de dispersão para amostras de cada cluster
for cluster in clusters:
    # obter índices de linha para amostras com este cluster
    row_ix = where(yhat == cluster)
    # criar dispersão dessas amostras
    plt.scatter(Y[row_ix, 0], Y[row_ix, 1],c=colors[cluster], label = 'Centroid ' + str(cluster+1))
# mostrar os gráficos
plt.legend()
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.title("KMeans Pacote Python")
plt.ylim([-1,26])
plt.show() 
############################################################

#!/usr/bin/python3                                                                                                                                                                                          
__version__ = "0"


###############################################################################                                                                                                                             
# Amin Boroomand                                                                                                                                                                       
#                                                                                                                                                                                              
#                                                                                                                                                                                                           
# GOAL: make clusters within communities and evaluate them                                                                                                                                                  
#                                                                                                                                                                                                           
# INPUTS:                                                                                                                                                                                                   
#                                                                                                                                                                                                           
# OUTPUTS:                                                                                                                                                                                                  
#                                                                                                                                                                                                           
# USAGE:                                                                                                                                                                                                    
#                                                                                                                                                                                                           
# This file import 2 files                                                                                                                                                                                  
# 1-admixture file contains testguid and 26 ethnicity                                                                                                                                                       
# 2-community assignment file contains testguid ID and the community ID                                                                                                                                     
# we join those file and then filter the data by community.                                                                                                                                                 
# we have about 337 community that we computed PCA for each community.                                                                                                                                      
###############################################################################                                                                                                                             


# imports                                                                                                                                                                                                   
from sklearn.cluster import DBSCAN
from sklearn import metrics
import sys,os,argparse, optparse
from random import shuffle
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import os
import pandas as pd
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt; plt.rcdefaults()
import time
from sklearn.cluster import AffinityPropagation
from itertools import cycle


Def importfile(commList,WDadrs='/DNAData/snapshots/ahna/communities_composition',birthfile="consented_birthYear_gender.2017-06-01.tsv" ,
ComAssignfile="consented_comm_assignments.2017-06-01.tsv", Admixturefile='consented_admixtures.2017-06-05.tsv'):
    """                                                                                                                                                                                                     
    GOAL: importfile imports the data files                                                                                                                                                                 
    INPUTS:                                                                                                                                                                                                 
        WDadrs: data directory                                                                                                                                                                              
        ComAssignfile: community assignment data file                                                                                                                                                       
        Admixturefile: admixture data file                                                                                                                                                                  
    OUTPUT:                                                                                                                                                                                                 
        returns people grouped by their community assignment                                                                                                                                                
    """
    print("IMPORT FILE")
    os.chdir(WDadrs)  #set the working directory                                                                                                                                                            
    ComAssignimport = pd.read_csv(ComAssignfile, sep='\t', header=None, names=['testGuid', 'communityId', 'normalized_score', 'parentID'])
    Birthfle=pd.read_csv(birthfile,sep='\t',header=None, names=['testGuid','age','gender'])
    Admixture = pd.read_csv(Admixturefile, sep='\t')  #import admixture dataset                                                                                                                             
    Admixture = Admixture. 
    fullfile = pd.merge(ComAssignimport, Admixture ,on='testGuid')  #join 2 files (the community assignment and ethnicity of each indv).                                                                    
    fullfile=pd.merge(fullfile,Birthfle ,on='testGuid')
    fullfile.drop(['gender','parentID'], axis=1, inplace=True)
    #fullfile['age']= [float(1998.-x)/87. for x in fullfile['age'].tolist()]                                                                                                                                
    #fullfile['normalized_score']=[(y-20.)/75. for y in fullfile['normalized_score'].tolist()]                                                                                                              
    groups = fullfile.groupby(by=['communityId'])  # groupby community ID                                                                                                                                   
    size=pd.DataFrame(groups.size())
    #CMcnt=pd.DataFrame(fullfile.groupby(by=['communityId']).count() , columns=['communityId','counts'])                                                                                                    
    #print(size, "\n\n\n", size.columns )                                                                                                                                                                   
    size = size.reset_index()
    CMcount=size[size['communityId'] == commList] # this results ins omething like 250 \t 341224 where 250 is the index and 341224 is the size                                                              
    groups = fullfile[(fullfile.communityId == commList) & (fullfile.normalized_score > 50)]   #ahna had this line                                                                                          
    #print(CMcount[0].item())                                                                                                                                                                               
    #savingpath="/DNAData5/amin/amin/results/"                                                                                                                                                              
    #x_ladan = fullfile[fullfile['communityId']==933 ].age.tolist()                                                                                                                                         
    #y_ladan = fullfile[fullfile['communityId']==933 ].normalized_score.tolist()                                                                                                                            
    #print (y_ladan)                                                                                                                                                                                        
    #plt.hexbin(x_ladan, y_ladan)                                                                                                                                                                           
    #plt.savefig(savingpath + "933NORMAL_AGE.png")                                                                                                                                                          
    return groups,size
def CMpiecharts (commList,ethnicityDict,admixtureThisCommDf,savingpath="/DNAData5/amin/amin/results/"):
    savingpath="/DNAData5/amin/amin/results/"
    x=admixtureThisCommDf.drop(labels=['communityId'],axis=1).mean()
    x[x>0.03]
    labelss = np.array([ethnicityDict[int(i)] for i in list(x.index)])
    xUse = x > 0.03
    admixturesToPlot = np.append(x[xUse].values,1-x[xUse].sum())
    labelsToPlot = np.append(labelss[xUse], 'trace')
    colors=('b', 'g', 'r', 'c', 'm', 'y', 'burlywood', 'w')
    plt.pie(admixturesToPlot, labels=labelsToPlot,autopct='%.0f%%',colors=colors)
    plt.axis("equal")
    plt.savefig(savingpath +str(commList)+ "CMpie.png")


    return


def MAXethn (admixtureThisCommDf):
    maxlist_colors=[]
    pure_admixture=admixtureThisCommDf.drop(labels=['communityId'],axis=1)
    for index, row in pure_admixture.iterrows():
        maxETHn=max(row)
        for individual_ethnicity in row:
            if individual_ethnicity == maxETHn:
                idec=row[row == maxETHn].index[0]
                maxlist_colors.append(int(idec))
    return maxlist_colors


def CMszpercent (size,groups,commList,prcnt=5):
    print("Computing the file size")
    sizes = size.reset_index()
    CMcount=sizes[sizes['communityId'] == commList]
    CMsize=CMcount[0].item()
    CMpercent=(prcnt*CMsize)/100
    return int(CMpercent),CMsize
def importCMnames (diraddrss="/DNAData/ahna/projects/map_exploration/data/",CMnamefile="community_branches.tsv" ):
    os.chdir(diraddrss)
    print("Importing community names")
    CMnamesfile=pd.read_csv(CMnamefile, sep='\t')
    CMid=list(CMnamesfile.id)
    CMnames=list(CMnamesfile.name)
    CMnamesDict=dict(zip(CMid,CMnames))
    return CMnamesDict




def ethnicitynames (diraddrss="/DNAData/snapshots/ahna/communities_composition/",ethnicitynamefile="ethnicity_regions.tsv" ):
    os.chdir(diraddrss)
    print("Importing ethnicity names")
    ethnctIDnames=pd.read_csv(ethnicitynamefile, sep='\t')
    ethnicityname=list(ethnctIDnames.name)
    regionId=list(ethnctIDnames.regionId)
    ethnicityDict=dict(zip(regionId,ethnicityname))
    return ethnicityDict
def runPCA(groups,commList,PCnumber,NWadrs="/DNAData5/amin/amin/amin_internship_2017"):
    start = time.time()
    print("Running PCA")




    #GOAL: runPCA runs a PCA on the people passed in features                                                                                                                                               
    #INPUTS: groups=data that cames from importfile function                                                                                                                                                
    #        PCnumber= number of PCs                                                                                                                                                                        
    #        comunitynumber= The community that we want to run PCA on                                                                                                                                       
    #OUTPUT: data= transform PC values to data                                                                                                                                                              
    #        comunitynumber= community name                                                                                                                                                                 


    os.chdir(NWadrs)
    pca = PCA(n_components=PCnumber)


#    for k, gp in groups:                                                                                                                                                                                   
#    print(k,type(k), comunitynumber,type(comunitynumber))                                                                                                                                                  
#        if k == commList:                                                                                                                                                                                  


#k = groups.get_group(commNumber); k.set_index('testGuid',inplace=True); k=k.drop(['age','normalized_score'],axis=1)                                                                                        
    m =groups.set_index('testGuid')


    admixtureThisCommDf=m.drop(['age','normalized_score'],axis=1)


 # print ('CommunityID=' + k)                                                                                                                                                                               
    clean = admixtureThisCommDf.columns[1:]  # take columns relevant to ethnicity for now                                                                                                                   
    cleanfile = admixtureThisCommDf[clean]
    values = cleanfile.values
    result = pca.fit(values)
    results = pca.explained_variance_ratio_
    #print(pca.explained_variance_ratio_)                                                                                                                                                                   
    data = pca.transform(values)
    end = time.time()
    print("PCA time",end - start)
    print("PCA time",end - start)
    #file=open("testpcafile.txt","a")                                                                                                                                                                       
    #file.write(results)                                                                                                                                                                                    
    #file.write('/n')                                                                                                                                                                                       
    #file.write(data)                                                                                                                                                                                       
    #file.close()                                                                                                                                                                                           
    np.savetxt(str(commList)+'filePC'+str(PCnumber), data, delimiter=",")
    return data, admixtureThisCommDf
def importPCA(commList,PCnumber,PCAfile,filadrs="/DNAData5/amin/amin/PCAfiles/"):


#def importPCA(filadrs="/DNAData5/amin/amin/PCAfiles/", PCAfile="file_2"):                                                                                                                                  
    os.chdir(filadrs)  #set the working directory                                                                                                                                                           
    L=str(commList)+'filePCageNRM'+str(PCnumber)
    data= np.genfromtxt (L, delimiter=",")


    #data = pd.read_csv(PCAfile, sep=',')                                                                                                                                                                   


    return data






def affinity (data):
    print("Compute affinity propagation")
    start = time.time()
    af = AffinityPropagation(affinity='euclidean', convergence_iter=50, copy=True,
          damping=0.99, max_iter=50, verbose=False).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    print(cluster_centers_indices)
    aflabels = af.labels_


    n_clusters_ = len(cluster_centers_indices)


    print('Estimated number of clusters: %d' % n_clusters_)
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))                                                                                                                           
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))                                                                                                                         
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))                                                                                                                               
    #print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))                                                                                                                 
    #print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))                                                                                                  
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, aflabels, metric='sqeuclidean'))
    Af_s_index= metrics.silhouette_score(data, aflabels, metric='sqeuclidean')
    end=time.time()
    print("Affinity time=", end-start)
    return n_clusters_,aflabels,cluster_centers_indices,Af_s_index


def affinPlot(Af_s_index,commList,data,n_clusters_,aflabels,cluster_centers_indices,savingpath="/DNAData5/amin/amin/results/"):
    plt.close('all')
    plt.figure(1)
    plt.clf()


    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = aflabels == k
        cluster_center = data[cluster_centers_indices[k]]
        plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
        for x in data[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)


    plt.title('Estimated number of clusters: %d' % n_clusters_+ "  S_index "+ str(Af_s_index))
    plt.savefig(savingpath +"CM"+str(commList)+ "Affiniti.png")
    return


def runKmeans(data, Ncluster=3):
    start1 = time.time()
    """                                                                                                                                                                                                     
    GOAL: runKmeans runs kmeans on ....                                                                                                                                                                     
    INPUTS: data= output of PCA                                                                                                                                                                             
            Ncluster= Number of clusters                                                                                                                                                                    
    OUTPUT: Kmeans= clustering lables                                                                                                                                                                       
    """
    KM = np.array(data)


    kmeans = KMeans(n_clusters=Ncluster, random_state=0).fit(KM)
    end1 = time.time()
    print("Kmean time",end1 - start1)
    return kmeans






def piecharts (ethnicityDict,kmeans,aflabels,admixtureThisCommDf,savingpath="/DNAData5/amin/amin/results/"):
    savingpath="/DNAData5/amin/amin/results/"
    uniqueLabels=np.unique(aflabels)
    #lablelist=[]                                                                                                                                                                                           
    for j in uniqueLabels:
        print("LABeLS",j)
        x=admixtureThisCommDf[aflabels==j].drop(labels=['communityId'],axis=1).mean()
        x[x>0.03]
        labelss = np.array([ethnicityDict[int(i)] for i in list(x.index)])
        xUse = x > 0.03
        admixturesToPlot = np.append(x[xUse].values,1-x[xUse].sum())
        labelsToPlot = np.append(labelss[xUse], 'trace')
        colors=('b', 'g', 'r', 'c', 'm', 'y', 'burlywood', 'w')
        print("labels of the cluster",labelsToPlot)
        plt.pie(admixturesToPlot, labels=labelsToPlot,autopct='%.0f%%',colors=colors)
        plt.axis("equal")
        plt.savefig(savingpath +"lable"+str(j)+ "NEWAffinityPIE.png")
        admixturesToPlot=None
        labelsToPlot=None
        labels=None
        #print(labelsToPlot)                                                                                                                                                                                


    return


#col1_Data=col1_Data.PCA                                                                                                                                                                                    
#col2_Data=col2_Data.PCA                                                                                                                                                                                    
#Kmeans=Kmeans.Kmean                                                                                                                                                                                        
#savingpath="/DNAData5/amin/amin/size_KM_"                                                                                                                                                                  




def 










    """                                                                                                                                                                                                     
    GOAL: plotPCA creates a scatter plot of two requested prinicipal compoments                                                                                                                             
    INPUTS:                                                                                                                                                                                                 
    OUTPUT:                                                                                                                                                                                                 
    """
    pcx=pcx-1
    pcy=pcy-1
    fs = 20
    col1_Data2 = data[:,pcx]
    col2_Data2 = data[:,pcy]
    #print("explained variance ratio",col1_Data2.explained_variance_ratio_)                                                                                                                                 
    with plt.style.context('seaborn-pastel'):
        #plt.figure(figsize=(8, 8))                                                                                                                                                                         
        #plt.scatter(col1_Data, col2_Data, s=20, alpha=.1, c=kmeans.labels_, linewidth=0)                                                                                                                   
        plt.hexbin(col1_Data2, col2_Data2)
        
        plt.legend(loc='lower center', fontsize=fs)
        plt.title('CMname: ' + commName+'  K is= '+str(bestK)+'  CMsize: '+ str(CoMMsize))
        plt.axis('equal')
       #plt.show()                                                                                                                                                                                          
        if savingpath is not None:
           plt.savefig(savingpath + str(commList) + "fUNC_PC_plot.png")
        return
def EthniPCAplot(maxlist_colors,commName,data,pcx=1,pcy=2,savingpath="/DNAData5/amin/amin/results/"):
    """                                                                                                                                                                                                    \
                                                                                                                                                                                                            
    GOAL: plotPCA creates a scatter plot of two requested prinicipal compoments                                                                                                                            \
                                                                                                                                                                                                            
    INPUTS:                                                                                                                                                                                                \
                                                                                                                                                                                                            
    OUTPUT:                                                                                                                                                                                                \
                                                                                                                                                                                                            
    """
    pcx=pcx-1
    pcy=pcy-1
    fs = 20
    col1_Data2 = data[:,pcx]
    col2_Data2 = data[:,pcy]
    with plt.style.context('seaborn-pastel'):
        #plt.figure(figsize=(8, 8))                                                                                                                                                                        \
                                                                                                                                                                                                            
        plt.scatter(col1_Data2, col2_Data2, s=20,c=maxlist_colors ,alpha=.1,  linewidth=0)                                                                                                                 \


        plt.xlabel('PC1', fontsize=fs)
        plt.ylabel('PC2', fontsize=fs)
        plt.legend(loc='lower center', fontsize=fs)
        plt.title('CMname: ' + str(commName))
        plt.axis('equal')
        plt.savefig(savingpath +"CM"+str(commName)+ "ETHNIpca.png")
        return








def Sindex (kmeans,data):
    '''                                                                                                                                                                                                     
    GOAL: compute Silhouette index. As closer to 1 the clustering number is better.                                                                                                                         
    inputs: kmeans=kmean lables which are the output of runKmeans function.                                                                                                                                 
            data=output of runPCA function.                                                                                                                                                                 
    outputs: S validation.                                                                                                                                                                                  
    '''
    labels = kmeans.labels_
    S=metrics.silhouette_score(data, labels, metric='euclidean')
    shuffledlabel=deepcopy(labels)
    shuffle(shuffledlabel)
    shuffledS=metrics.silhouette_score(data,shuffledlabel, metric='euclidean')
    return S, shuffledS


def runDBscan (data,kmeans_best,EP,sampmin ):
    start2=time.time()
    """                                                                                                                                                                                                     
    GOAL:                                                                                                                                                                                                   
    INPUTS:                                                                                                                                                                                                 
    OUTPUT:                                                                                                                                                                                                 
    """
    db = DBSCAN(eps=EP, min_samples=sampmin, metric='manhattan').fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    # Number of clusters in labels, ignoring noise if present.                                                                                                                                              
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    DB_cluster= n_clusters_
    if len(set(labels)) >1 :


        DB_S= metrics.silhouette_score(data, labels)
    else:
        DB_S=-2
    #print("In comparison with K-mean lables when K="+ str(Ncluster))                                                                                                                                       
    #Homogeneity=metrics.homogeneity_score(kmeans_best.labels_, labels)                                                                                                                                     
    #Completeness=metrics.completeness_score(kmeans_best.labels_, labels)                                                                                                                                   
    #v_measure= metrics.v_measure_score(kmeans_best.labels_, labels)                                                                                                                                        
    #Adjusted_Rand_Index= metrics.adjusted_rand_score(kmeans_best.labels_, labels)                                                                                                                          
    #Adjusted_Mutual_Information=metrics.adjusted_mutual_info_score(kmeans_best.labels_, labels)                                                                                                            
    dblabels=labels
    end2 = time.time()
    #print("DBscan time",end2 - start2)                                                                                                                                                                     
    return dblabels,core_samples_mask,n_clusters_,DB_S


def DBscatterPlt (EPbest,Best_DBs,labels,data,core_samples_mask,DBn_clusters_,savingpath="/DNAData5/amin/amin/results/"):
    """                                                                                                                                                                                                    \
                                                                                                                                                                                                            
    GOAL:                                                                                                                                                                                                   
    INPUTS:                                                                                                                                                                                                \
    OUTPUT:                                                                                                                                                                                                \
                                                                                                                                                                                                            
    """
    n_clusters_=DBn_clusters_
# Black removed and is used for noise instead.                                                                                                                                                              
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors = ['r', 'g', 'b']
    for k, col in zip(unique_labels, colors):




        class_member_mask = (labels == k)


        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o',  color=col, markersize=4.5,alpha=0.1,linewidth=0)


        # plot unclustered points                                                                                                                                                                           
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k', #tuple(col),                                                                                                                                 
                 markeredgecolor='k', markersize=4.5,alpha=1.0,linewidth=0)
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.axis('equal')
    plt.title('number of clusters: %d' % n_clusters_ +"  best EP(max dist within a cluster)="+" %.2f"% EPbest  +"   S index =" + "%.2f" %Best_DBs )
    if savingpath is not None:
        plt.savefig(savingpath + str(n_clusters_) + "DBscan.png")


def generalDBscatterPlt (PCnumber,commList,samplemin,DB_Sindx,DBpar,labelsss,data,core_samples_maskss,DBn_clusters_ss,savingpath="/DNAData5/amin/amin/results/"):
    n_clusters_=DBn_clusters_ss
    unique_labelsss = set(labelsss)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labelsss))]
    colors = ['r', 'g', 'b']
    for k, col in zip(unique_labelsss, colors):




        class_member_maskss = (labelsss == k)


        xy = data[class_member_maskss & core_samples_maskss]
        plt.plot(xy[:, 0], xy[:, 1], 'o',  color=col, markersize=4.5,alpha=0.1,linewidth=0)


        # plot unclustered points                                                                                                                                                                          \
                                                                                                                                                                                                            
        xy = data[class_member_maskss & ~core_samples_maskss]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k', #tuple(col),                                                                                                                                \
                                                                                                                                                                                                            
                 markeredgecolor='k', markersize=4.5,alpha=1.0,linewidth=0)
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.axis('equal')
    plt.title(str(commList)+"sample Min "+str(samplemin)+ "DB_Sindex"+str(DB_Sindx)+"EP"+str(DBpar)+ "Ncluster"+str(n_clusters_) )
    if savingpath is not None:
        plt.savefig(savingpath +str(commList)+"smpmin"+str(samplemin)+"EP"+str(DBpar) +"PCn"+str(PCnumber) +"appendix.png")








def DBbarplot(DBeps,DBsList,Best_DBs ,commList,PCnumber, savingpath="/DNAData5/amin/amin/results/"):
    """                                                                                                                                                                                                    \
                                                                                                                                                                                                            
    GOAL: DBbarplot  creates a bar chart of validity metrics of DBscan                                                                                                                                     \
                                                                                                                                                                                                            
    INPUTS:clusternumberlist = number of clusters , validationlist=validation number associated to each number of cluster                                                                                  \
                                                                                                                                                                                                            
    OUTPUT: validation plot                                                                                                                                                                                \
                                                                                                                                                                                                            
    """
    y_pos = np.arange(len(DBeps))
    plt.bar(y_pos,DBsList, align='center', alpha=0.5)
    plt.xticks(y_pos,DBeps)
    plt.ylabel('Silhouette index')
    plt.xlabel('EP')
    plt.ylim((-1,1))
    plt.title("CM num=" + str(commList) + " PC=" + str(PCnumber)+ "  best DB_S="+"%.2f"% Best_DBs ) # str(Best_DBs))                                                                                        
    if savingpath is not None:
        plt.savefig(savingpath +"_commnum" +str(commList)+"_PC"+str(PCnumber) + "DBvalidation.png")
    return




def plotValidityMetrics(Best_DBs,DBn_clusters_,Svalue_best,klist,validationlist,commList,PCnumber,shuffledlist, savingpath="/DNAData5/amin/amin/results/"):
    """                                                                                                                                                                                                     
    GOAL: plotValidityMetrics creates a bar chart of validity metrics                                                                                                                                       
    INPUTS:clusternumberlist = number of clusters , validationlist=validation number associated to each number of cluster                                                                                   
    OUTPUT: validation plot                                                                                                                                                                                 
    """
    y_pos = np.arange(len(klist))
    plt.bar(y_pos,validationlist, align='center', alpha=0.5)
    plt.xticks(y_pos, klist)
    plt.ylabel('Silhouette index')
    plt.xlabel('Number of clusters')
    plt.ylim((-1,1))
    plt.title("CM num=" + str(commList) + " PC=" + str(PCnumber)+ "  Best K_S ="+"%.2f"% Svalue_best+ "   DB_S="+ "%.2f" %Best_DBs)
    for j in range(len(klist)):
        b=klist[j]
        c=shuffledlist[j]
        plt.plot([float(b)-0.5-min(klist), float(b)+0.5-min(klist)], [c, c], 'k--')
    plt.plot([float(DBn_clusters_)-0.5-(DBn_clusters_), float(DBn_clusters_)+0.5-(DBn_clusters_)], [Best_DBs, Best_DBs])
    if savingpath is not None:
        plt.savefig(savingpath +"_commnum" +str(commList)+"_PC"+str(PCnumber) + "validation.png")
    return




def runPipeline(commList,NcomponentList,klist,DBeps,SMPmin):
    start3 = time.time()
    """                                                                                                                                                                                                     
    GOAL:                                                                                                                                                                                                   
    INPUTS:                                                                                                                                                                                                 
    OUTPUT:                                                                                                                                                                                                 
    """
    groups,size = importfile(commList,WDadrs='/DNAData/snapshots/ahna/communities_composition',birthfile="consented_birthYear_gender.2017-06-01.tsv"  ,ComAssignfile="consented_comm_assignments.2017-06-01\
.tsv", Admixturefile='consented_admixtures.2017-06-05.tsv')
    CMdict=importCMnames (diraddrss="/DNAData/ahna/projects/map_exploration/data/",CMnamefile="community_branches.tsv" )
    commName=CMdict[commList]
    ethnicityDict=ethnicitynames(diraddrss="/DNAData/snapshots/ahna/communities_composition",ethnicitynamefile="ethnicity_regions.tsv" )
    #CMsizePrcnt_orgn=CMszpercent(groups,commList,prcnt=5)                                                                                                                                                  
    #CMsizePrcnt=CMsizePrcnt_orgn[0]                                                                                                                                                                        
    #CoMMsize=CMsizePrcnt_orgn[1]                                                                                                                                                                           
    col1_Data = []
    col2_Data = []
    CMsizePrcnt,CoMMsize=CMszpercent(size,groups,commList,prcnt=5)
    PCnumber = NcomponentList
    validationlist=[]
    shuffledlist=[]
    data,admixtureThisCommDf=runPCA(groups,commList,PCnumber,NWadrs="/DNAData5/amin/amin/amin_internship_2017")
    #data= importPCA(commList,PCnumber,filadrs="/DNAData5/amin/amin/PCAfiles/", PCAfile=str(commList)+'filePCageNRM'+str(PCnumber))                                                                         
    n_clusters_,aflabels,cluster_centers_indices,Af_s_index=affinity(data)
    afPlot=affinPlot(Af_s_index,commList,data,n_clusters_,aflabels,cluster_centers_indices,savingpath="/DNAData5/amin/amin/results/")
    maxlist_colors=MAXethn (admixtureThisCommDf)
    ethnicityplot=EthniPCAplot(maxlist_colors,commName,data,pcx=1,pcy=2,savingpath="/DNAData5/amin/amin/results")
    CMpie=CMpiecharts (commList,ethnicityDict,admixtureThisCommDf,savingpath="/DNAData5/amin/amin/results/")
    savingpath="/DNAData5/amin/amin/results/"


    for Ncluster in klist:
        kmeans = runKmeans(data, Ncluster)
        Svalue=Sindex(kmeans,data)
        piematerial=piecharts (ethnicityDict,kmeans,aflabels,admixtureThisCommDf,savingpath="/DNAData5/amin/amin/results/")
    #pcx=0                                                                                                                                                                                                  
    #pcy=1                                                                                                                                                                                                  
    #fs=20                                                                                                                                                                                                  
    #for PCs in data:                                                                                                                                                                                       
        #col1_Data.append(PCs[pcx])                                                                                                                                                                         
        #col2_Data.append(PCs[pcy])                                                                                                                                                                         
    #with plt.style.context('seaborn-pastel'):                                                                                                                                                              
        #plt.figure(figsize=(8, 8))                                                                                                                                                                        \
        #plt.scatter(col1_Data, col2_Data, s=20, alpha=.1, c=kmeans.labels_, linewidth=0)                                                                                                                  \
        #plt.hexbin(col1_Data, col2_Data)                                                                                                                                                                   
        #plt.xlabel('PC1', fontsize=fs)                                                                                                                                                                     
        #plt.ylabel('PC2', fontsize=fs)                                                                                                                                                                     
        #plt.legend(loc='lower center', fontsize=fs)                                                                                                                                                        
        #plt.title('CMname: ' + commName+'  CMsize: '+ str(CoMMsize))                                                                                                                                       
        #plt.axis('equal')                                                                                                                                                                                  
        #plt.savefig(savingpath + str(commList) + "AAGGEEheattt.png")                                                                                                                                       
        #return                                                                                                                                                                                             
    for Ncluster in klist:
        kmeans = runKmeans(data, Ncluster)
        Svalue=Sindex(kmeans,data)
        #piematerial=piecharts (kmeans,admixtureThisCommDf)                                                                                                                                                 
        #clusternumberlist.append(Ncluster)                                                                                                                                                                 
        validationlist.append(Svalue[0])
        shuffledlist.append(Svalue[1])
        Svalue_best = -1
        if Svalue[0] > Svalue_best:


        if Svalue[0] > Svalue_best:
            Svalue_best=Svalue[0]
            kmeans_best = kmeans
            bestK=Ncluster
            plotresult = plotPCA(CoMMsize,commName,bestK,data,commList,kmeans,pcx=1,pcy=2,savingpath="/DNAData5/amin/amin/results")
    for samplemin in SMPmin:
        PCnumber = NcomponentList
        clusternumberlist=[]
        DBsList=[]
        #runPCAresult = runPCA(groups,commList,PCnumber)                                                                                                                                                    
        #data = runPCAresult[0]                                                                                                                                                                             
        #comunitynumber = runPCAresult[1]                                                                                                                                                                   
        #data=runPCA(groups,commList,PCnumber,NWadrs="/DNAData5/amin/amin/amin_internship_2017")                                                                                                            
        Best_DBs=-3
        print("The best S index of K_mean",Svalue_best)
        for DBpar in DBeps:
            DBscan= runDBscan (data,kmeans_best,EP=DBpar,sampmin=samplemin )
            DB_Sindx=DBscan[3]
            DBsList.append(DB_Sindx)
            n=1
            savepath="/DNAData5/amin/amin/results/"
            if DB_Sindx> Best_DBs:
                Best_DBs=DB_Sindx
                labels=DBscan[0]
                core_samples_mask=DBscan[1]
                DBn_clusters_=DBscan[2]
                EPbest=DBpar
            if DB_Sindx > 0:
                labelsss=DBscan[0]
                core_samples_maskss=DBscan[1]
                DBn_clusters_ss=DBscan[2]
                print ('DBS',str(DB_Sindx), "EP=",str(DBpar), "Samplemin=", samplemin,"CLnum",DBn_clusters_ss, "PC", PCnumber)
                n=n+1
                plt.figure(figsize=(10,10))
                plt.subplot(2,2,n);
                gDBscnSCTtplt=generalDBscatterPlt (PCnumber,commList,samplemin,DB_Sindx,DBpar,labelsss,data,core_samples_maskss,DBn_clusters_ss)
                plt.tight_layout()
                #plt.savefig(savepath+str(commList)+"_"+commName +"P_PC"+str(PCnumber) + "Minsmp_"+str(samplemin) +"EP"+str(DBpar)+"appendix.pdf")                                                          
        print("Best value of DB_scan S_index  was", Best_DBs, "\n\n")
        DBplot=DBscatterPlt (EPbest,Best_DBs,labels,data,core_samples_mask,DBn_clusters_)
        plotCommunitySummary(samplemin,CoMMsize,commName,DBeps,DBsList,EPbest,Best_DBs,DBn_clusters_,core_samples_mask,labels,Svalue_best,bestK,klist,validationlist,PCnumber,shuffledlist, data, kmeans_be\
st, commList, savingpath="/DNAData5/amin/amin/results/")
    end3 = time.time()
    #print("pipeline  time",end3 - start3)                                                                                                                                                                  
    return


                #print("community name=",communitynumber, "number of components=",PCnumber,"S index=",Svalue)                                                                                               




def plotCommunitySummary(samplemin,CoMMsize,commName,DBeps,DBsList,EPbest,Best_DBs,DBn_clusters_,core_samples_mask,labels,Svalue_best,bestK,klist,validationlist,PCnumber,shuffledlist, data, kmeans_best, \
commList, savingpath="/DNAData5/amin/amin/results/"):
    """                                                                                                                                                                                                     
    GOAL:                                                                                                                                                                                                   
    INPUTS:                                                                                                                                                                                                 
    OUTPUT:                                                                                                                                                                                                 
    """
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1);
    XplotValidityMetrics= plotValidityMetrics(Best_DBs,DBn_clusters_,Svalue_best,klist,validationlist,commList,PCnumber,shuffledlist, savingpath=None)
    plt.subplot(2,2,2);
    plotresult = plotPCA(CoMMsize,commName,bestK,data,commList,kmeans=kmeans_best,pcx=1,pcy=2,savingpath=None)
    plt.subplot(2,2,3);
    DBbplot= DBbarplot(DBeps,DBsList,Best_DBs ,commList,PCnumber, savingpath=None)
    plt.subplot(2,2,4);
    XDBscatterplt=DBscatterPlt(EPbest,Best_DBs,labels,data,core_samples_mask,DBn_clusters_,savingpath=None)
    plt.tight_layout()
    plt.savefig(savingpath+str(commList)+"_"+commName +"_PC"+str(PCnumber) + " Min "+str(samplemin) +"_K-DB-validation.pdf")




def main(commList,NcomponentList):
    klist = [2]#,3,4]                                                                                                                                                                                       
    DBeps=[0.00001]#, 0.001,0.005,0.01,0.1,0.2,0.3,0.4,0.5, 1,10,100]                                                                                                                                       
    SMPmin=[2]#,3,4,5,10,20,30,50,150,350,500,1000]                                                                                                                                                         
    return runPipeline(commList,NcomponentList,klist,DBeps,SMPmin)


#main()                                                                                                                                                                                                     




if __name__ == "__main__":


    PROGRAM = os.path.basename(sys.argv[0])
    VERSION = __version__
    USAGE = "python3 /DNAData5/amin/amin/amin_internship_2017/community_clusters.py  -i <community nubmers>  -d <PC numbers>  " # -s <number of clusters>                                                   
    SYNOPSIS = "it perform PCA, perform K-mean clustering and DBscan and then it evaluate those clusterings "
    parser = optparse.OptionParser(usage="%s %s"%(PROGRAM,USAGE), version="%s version %s"%(PROGRAM,VERSION), description=SYNOPSIS)
    parser.add_option("-i", dest="commList", type="string", metavar="Filename", action="store", default=False, help="community numbers(name of the community)")
    parser.add_option("-d", dest="NcomponentList", type="string", metavar="Filename", action="store", default=False, help="number of PCs")
    #parser.add_option("-s", dest="klist", type="string", metavar="Filename", action="store", default=False, help="number of clusters")                                                                     
    (options, args) = parser.parse_args()
    commList = int(options.commList)
    NcomponentList = int(options.NcomponentList)
    #klist = int(options.klist)                                                                                                                                                                             
    #main(commList, klist,NcomponentList)                                                                                                                                                                   
    main(commList,NcomponentList)
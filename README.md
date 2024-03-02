# VG_for_arrhythmia_classification_with_GCN

# Summary

- [VG\_for\_arrhythmia\_classification\_with\_GCN](#vg_for_arrhythmia_classification_with_gcn)
- [Summary](#summary)
- [Overview](#overview)
- [System Requirements](#system-requirements)
  - [Clone](#clone)
  - [Packages](#packages)
  - [Environments](#environments)
- [Dataset](#dataset)
- [Reproduce](#reproduce)
  - [VG Folder](#vg-folder)
  - [VVG Folder](#vvg-folder)


# Overview

This repository showcases the implementation of experiments detailed in the article [Leveraging Visibility Graphs for Enhanced Arrhythmia Classification](test). The research stems from my Master's thesis in Computer Science at the Federal University of Ouro Preto. It delves into the representation of ECG signals as graphs and the classification of arrhythmias with [GCN](https://tkipf.github.io/graph-convolutional-networks/) (Graph Convolutional Network). 

The code was written by Rafael Oliveira. Last updated: 2024.03.01.

# System Requirements


## Clone

```shell
git clone https://github.com/raffoliveira/VG_for_arrhythmia_classification_with_GCN.git
```

```shell
cd VG_for_arrhythmia_classification_with_GCN
```

## Packages

The main packages are listed below. 

+ dgl==2.0.0
+ igraph==0.11.4
+ matplotlib==3.8.3
+ networkx==3.2.1
+ numpy==1.26.4
+ pandas==2.2.0
+ scipy==1.12.0
+ seaborn==0.13.2
+ torch==2.2.0
+ ts2vg==1.2.3


To install all the packages described in the requirements file using below command.

```shell
pip install -r requirements.txt
```

## Environments

The development version of the packages has undergone testing on Linux. The system and drivers information are listed below.

+ Linux Mint 20.1 
+ CPU AMD Ryzen Threadripper 3960X 24-Core 3.8GHz
+ GPU NVIDIA GeForce RTX 3090
+ CUDA 11.1

# Dataset

The dataset utilized originates from the [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) (The Massachusetts Institute of Technology - Beth Israel Hospital Arrhythmia Database) and is accessible in the `Data` directory. The dataset is divided into two subsets, **Train** and **Test**, following the methodology proposed by [De Chazal et al.](https://www.researchgate.net/publication/8459885_Automatic_Classification_of_Heartbeats_Using_ECG_Morphology_and_Heartbeat_Interval_Features) The ECG signals are labeled numerically, corresponding to individual patients, and are provided in `.mat` format for ease of use within the code.


# Reproduce

The visibility graph (VG) and vector visibility graph (VVG) were elucidated in another GitHub project of mine, [VG_VVG_implementation](https://github.com/raffoliveira/VG_VVG_implementation). Notably, the implementation outlined in that project is employed throughout all experiments presented here.

## VG Folder

This folder comprises five experiments about visibility graph, each described below:

+ New Architectures: This experiment explores five Graph Convolutional Networks (GCNs) and three Convolutional Neural Networks (CNNs) architectures. The inclusion of CNNs serves as a comparative baseline for evaluating arrhythmia classification performance across different architectures.
  
+ Segmentation: Delving into the influence of segmentation width, measured by the number of points per heartbeat, this experiment assesses its impact on the effectiveness of automatic arrhythmia classification architectures.
  
+ Aggregation: This experiment aims to evaluate the effect of information aggregation on GCN architectures, particularly focusing on extrinsic graph data. GCNs are designed to learn from the inherent structures of graphs used during training.
  
+ Swap Datasets: Investigating the impact of reversing datasets—using the test set for training and the training set for testing—this experiment seeks to determine whether it can enhance the performance of GCN and CNN architectures.
  
+ Intra-patient: here, the intra-patient paradigm is explored as a contrast to the inter-patient approach. Unlike the inter-patient paradigm, which involves no overlap of data from the same patient between training and testing sets, the intra-patient approach allows for the presence of heartbeat data from the same patient in both datasets.

To execute each file, navigate to the desired folder and simply run the following command.

```
python file_name.py
```

## VVG Folder

This folder comprises three experiments focusing on the vector visibility graph (VVG) method. The descriptions provided for each experiment remain applicable here.

Given the considerable computational demand of the VVG method, execution is split into two stages:

1. **Training**: In this stage, the GCN network undergoes training, and the resulting model is saved in the `Models` folder.

2. **Testing**: Once the model is saved, it can be loaded for testing purposes.

Before executing each file, the user should set the `MODE=Train` or `MODE=Test` variable in the main method to specify the desired stage. Subsequently, the command below can be executed to run the process.

```
python file_name.py
```


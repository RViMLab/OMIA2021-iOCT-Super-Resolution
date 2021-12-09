

# Intra-operative OCT (iOCT) Image Quality Enhancement: A Super-Resolution Approach using High Quality iOCT 3D Scans

**Presented at Ophthalmic Medical Image Analysis (OMIA) 2021, MICCAI WORKSHOP**:  [Intra-operative OCT (iOCT) Image Quality Enhancement](https://link.springer.com/chapter/10.1007/978-3-030-87000-3_3)

## Introduction 

This reposirtory contains the implementation of the methods presented in the paper "Intra-operative OCT (iOCT) Image Quality Enhancement: A Super-Resolution Approach using High Quality iOCT 3D Scans", presented at OMIA 2021.

## Abstract

Effective treatment of degenerative retinal diseases will require robot-assisted intraretinal therapy delivery supported by excellent
retinal layer visualisation capabilities. Intra-operative Optical Coherence
Tomography (iOCT) is an imaging modality which provides real-time,
cross-sectional retinal images partially allowing visualisation of the layers where the sight restoring treatments should be delivered. Unfortunately, iOCT systems sacrifice image quality for high frame rates, making
the identification of pertinent layers challenging. This paper proposes a
Super-Resolution pipeline to enhance the quality of iOCT images leveraging information from iOCT 3D cube scans.We first explore whether 3D
iOCT cube scans can indeed be used as high-resolution (HR) images by
performing Image Quality Assessment. Then, we apply non-rigid image
registration to generate partially aligned pairs, and we carry out data
augmentation to increase the available training data. Finally, we use
CycleGAN to transfer the quality between LR and HR domain. Quantitative analysis demonstrates that iOCT quality increases with statistical
significance, but a qualitative study with expert clinicians is inconclusive
with regards to their preferences.

## Methods

### 1. Dataset
We used an internal dataset of intra-operative retinal surgery videos and OCT/iOCT scans from 66 patients acquired at Moorfields Eye Hospital, London, UK. 
We ended up having 983 images per type (**V-iOCT**, **C-iOCT**, **C-pOCT**). 


### 2. Image Quality Assessment (IQA)
IQA metrics implementation files can be found in ``` metrics ```. L_feat, FID, NIQE and GCF.

### 3. Registration
Registration implementation files can be found in ``` registration ```. 



### 4. Data Augmentation

### 5. Super Resolution using CycleGAN and Pix2Pix

## 6. Results

### IQA

### Quantitative Analysis

<img src="/imgs/table_1.png">



asdasdad









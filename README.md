Objective: 
Fine tune an existing model to the BUS-BRA dataset and deploy it as an endpoint

Data Exploration:

Verified 1064 cases with with majority (811) cases having 2 images (left and right)
No NULL entries found
Malignant label is roughly 1/3 the frequency of benign

BIRADs spread across labels:
 Pathology  benign  malignant
BIRADS                      
2             552         10
3             432         31
4             280        413
5               4        153

Class presence among different machines used is well split:
 Pathology                     benign  malignant
Device                                         
GE Logiq 5 @10-12MHz             566        243
GE Logiq 7 @10-14MHz             617        285
Toshiba Aplio 300 @12-14 MHz      71         68
U-Systems @10-14MHz               14         11

Data Splits:
We will use HOP assignment that is part of this dataset to split our data
HOP -> 1 means it is test set
HOP -> 2 means it is part of our train and validation set

The split folders will be created prior to any training:
pool 1489 {'benign': 0.677, 'malignant': 0.323} # used for validation too
test 386 {'benign': 0.674, 'malignant': 0.326} # used strictly for testing

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

Only 4 samples seem to have missmatches dimensions:
    size mismatches: 4 [
    ('bus_0856-l', (387, 353), (387, 405)), 
    ('bus_0857-r', (384, 393), (384, 409)), 
    ('bus_0912-l', (360, 352), (360, 354)), 
    ('bus_0990-l', (540, 352), (540, 354))]
Their associated BBOXs
    bus_0856-l | bbox: [46,29,312,176] | actual: (387, 353) | in test: True
    bus_0857-r | bbox: [28,3,296,217] | actual: (384, 393) | in test: False
    bus_0912-l | bbox: [47,96,301,177] | actual: (360, 352) | in test: False
    bus_0990-l | bbox: [119,54,298,176] | actual: (540, 352) | in test: True

Model Choice: RadImageNet
Credit to: Mei, Xueyan and Liu, Zelong and Robson, Philip M. and Marinelli, Brett and Huang, Mingqian and Doshi, Amish and Jacobi, Adam and Cao, Chendi and Link, Katherine E. and Yang, Thomas and Wang, Ying and Greenspan, Hayit and Deyer, Timothy and Fayad, Zahi A. and Yang, Yang

Noticed that there are crosshairs and text labels on data samples:
    Seemingly similar on both benign and malignant cases -> crosshairs used to identify the lesion

The plan is to train a baseline model with the current dataset, if we notice performance drop post augmentation or by scanner then we will re-asses
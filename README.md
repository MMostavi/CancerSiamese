# CancerSiamese
## Codes for CancerSiamese paper

In this repo, you find codes for CancerSiamese, a one shot learning approach for classifying primary and metastatic tumors with only one sample. The primary and metastatic samples are obtained from TCGA and MET500 datasets, respectively. 

Link to TCGA:
https://portal.gdc.cancer.gov/

Link to MET500:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5995337/

## Key contributions
- To the best of our knowledge, this is the only model that considers only one gene expression sample for classifying primary or metastatic tumors.
- Analyzing primary and metastatic tumors in parallel 
- Extracting common cancer markers represented among primary and metastatic tumors with their corresponding functions. 

During training model takes in two paired gene expresison inputs which can be either from same or different cancer type. Once the model is trained, it will get pairs of query and support sample and make one shot task.
![image](https://user-images.githubusercontent.com/22861849/95668520-bc4b6b00-0b3a-11eb-865f-52714854acb4.png)

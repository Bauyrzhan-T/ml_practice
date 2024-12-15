# Prediction of lung-cancer based on gene expression (GE) data

Algorithm structure:
  1. Fetching training data from NCBI GEO.
  2. Normalization.
  3. Feature selection (ANOVA F-test)
  4. Classification via Random Forest.
  5. Cross-validation (5-fold).
  6. Testing on new data.

Training and testing peformed with data obtained from NCBI GEO public database:
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=gse75037
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1012828

Notes:
  1. Algorithms assumes the same platoform has been used for gene expression study (in our case, GPL6884	Illumina HumanWG-6 v3.0 expression beadchip). Cross-platform analysis will require additional modifications to the code.
  2. As for now, the cancer-positive and cancer-negative GE test samples have to be specified manually (line 24-29), with reference to the information provided by GEO.

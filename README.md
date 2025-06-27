# GCNMF-SDA:Predicting snoRNA-disease associations based on graph convolution and non-negative matrix factorization

## Requirements
* ### dgl == 2.0.0
* ### python == 3.9
* ### torch == 2.2.2
* ### numpy == 1.24.4
* ### pandas == 2.2.3
* ### scikit-learn == 1.1.3

## <span style="color:red">data</span>

* ### ass.txt: snoRNA and disease association data
* ### adj_index.csv：snoRNA and disease association matrix
* ### GIPK-d.np and GIPK-s.np :GIPK similarity of snoRNA and disease

* ### sno_p2p-smith.csv:Sequence similarity of snoRNA

* ### sno_d2d_do.csv and sno_mesh_do.csv:Two kinds of disease semantic similarity; disease_similarity.csv:Semantic Similarity of Comprehensive

* ### s_fusion.csv: snoRNA similarity after using SNF fusion

* ### d_fusion.csv: disease similarity after using SNF fusion

## <span style="color:red">code </span>

###  utils.py
- Methods for data processing.

---

###  main.py
- Run the GCNMF-SDA model.

####  Load Data
- Files: `s_fusion.csv`, `d_fusion.csv`, `adj_index.csv`, and `ass.txt`.

####  Feature Extraction
- `get_low_feature()`: SnoRNA extraction and disease characterization using NMF.
- `extract_gcn_features()`: SnoRNA extraction and disease characterization using GCN.

####  Negative Sample Selection
- `samples_choose()`: Constructs balanced datasets by selecting negative samples corresponding to positive samples.

####  5-Fold Cross Validation
- `StratifiedKFold()`: Splits the balanced dataset evenly into 5 folds.
- `EarlyStoppingAUC()`: AUC-based early stopping strategy and preserving model weights.



---
### predict.py
- Importing model weights for prediction.
-  `checkpoint.pt`: weights of the trained model.
- `model.load_state_dict(torch.load('checkpoint.pt'))`: Loads the trained model weights for prediction.
###  DEG.py
- DEG analysis for unvalidated snoRNAs.

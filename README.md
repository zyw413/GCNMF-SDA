# GCNMF-SDA:Predicting snoRNA-disease associations based on graph convolution and non-negative matrix factorization

## Requirements
* ### dgl == 2.0.0
* ### python == 3.9
* ### torch == 2.2.2
* ### numpy == 1.24.4
* ### pandas == 2.2.3
* ### scikit-learn == 1.1.3

## <span style="color:red">data</span>

* ### ass.txt: snoRNA and disease association data>

* ### GIPK-d.np and GIPK-s.np :GIPK similarity of snoRNA and disease

* ### sno_p2p-smith.csv:Sequence similarity of snoRNA

* ### sno_d2d_do.csv and sno_mesh_do.csv:Two kinds of disease semantic similarity; disease_similarity.csv:Semantic Similarity of Comprehensive

* ### s_fusion.csv: snoRNA similarity after using SNF fusion

* ### d_fusion.csv: disease similarity after using SNF fusion

## <span style="color:red">code </span>

* ### utils.py: Methods of data processing

* ### main.py: run GCNMF-SDA model

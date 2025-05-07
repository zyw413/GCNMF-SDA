# GCNMF-SDA:Predicting snoRNA-disease associations based on graph convolution and non-negative matrix factorization
![GCNMF-SDA](GCNMF-SDA.png)

## data

### ass.txt:
### GIPK-d.np and GIPK-s.np :GIPK similarity of snoRNA and disease

### sno_p2p-smith.csv:Sequence similarity of snoRNA

### sno_d2d_do.csv and sno_mesh_do.csv:Two kinds of disease semantic similarity; disease_similarity.csv:Semantic Similarity of Comprehensive

### s_fusion.csv: snoRNA similarity after using SNF fusion

### d_fusion.csv: disease similarity after using SNF fusion

## code 
### utils.py: Methods of data processing
### main.py: run GCNMF-SDA model

# 150k Python Dataset

We provide source file used to obtain the py150 dataset (https://www.sri.inf.ethz.ch/py150). The archive contains the following files:
- data.tar.gz -- Archive containing all the source files
- python100k_train.txt -- List of files used in the training dataset.
- python50k_eval.txt -- List of files used in the evaluation dataset.
- github_repos.txt -- List of GitHub repositories and their revisions used to obtain the dataset.

Note that the order of python100k_train.txt and python100k_train.json (containing the ASTs of the parsed files) are the same.
That is, parsing the n-th file from python100k_train.txt produces n-th ASTs in python100k_train.json


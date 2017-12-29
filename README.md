Reinforcement Learning for Relation Classification from Noisy Data
==========
Relation classification from noisy data, aiming to categorize semantic relations between two entities given a plain text with the automantically generated training data. previous studies adopt multi-instance learning to consider the noises of instances and can not handle the sentence-level prediction. In this work, we propose a new model for relation classification, which consists of an instance selector and a relation classifier. This formalization enables our model to extract relations at the sentence level from noisy data. We provide the source code and datasets of the AAAI 2018 paper: "Reinforcement Learning for Relation Classification from Noisy Data".


DataSets
=========
We provide dataset in data folder. The data is download from [[data]](https://github.com/thunlp/NRE/blob/master/data/data.zip). They preprocess the original data to make it satisfy the input format of the codes. The data is originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text.". [[Download]](http://iesl.cs.umass.edu/riedel/ecml/)

To run out code, the dataset should be put in the data folder. There're two sub-folders pretrain/ and RE/ and a file vec.bin in the data/ folder.

Codes
=========
We publish the codes of "Reinforcement Learning for Relation Classification from Noisy Data" here.
We refer to the implement code of NRE model published at [[code]](https://github.com/thunlp/NRE).

Compile
=========
Just type "make" in the corresponding folder.

Train
========
For training, you need to type "./main [method] [alpha]" in the corresponding folder.

The output of the model will be saved in folder result/.

Parameter Setting:
+ method: current training process. "rlpre" measn pretrain the instance selector. "rl" means jointly train the instance selector and relation classifier.
+ alpha: learning rate

Test
========
For test, you need to type "./main test" in the corresponding folder.


Cite
=========
If you use the code, pleasee cite the following paper:
[Feng et al. 2016] Jun Feng, Minlie Huang, Li Zhao, Yang Yang, and Xiaoyan Zhu. Reinforcement Learning for Relation Classification from Noisy Data. In AAAI2018. [[pdf]](http://aihuang.org/static/papers/AAAI2018Denoising.pdf)

Reference
=========
[1] [Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.

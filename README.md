# NRN

The official implementation of the paper: Knowledge Graph Reasoning over Entities and Numerical Values

## Update on raw data

We have added the raw data for constructing the KGs in to <code>./data/</code>. 

The raw data we use is from the following github repository, if you use the data, please also cite their paper:
[https://github.com/mniepert/mmkb](https://github.com/mniepert/mmkb)


## Download of processed data
The input files for the models can be downloaded here. After decompressing, they should be put in the root directory of this repository. The are two directories in the zip file, and the small one serves for debuging and unit testing. They can be downloaded from 
[here](https://drive.google.com/file/d/1QTX1i5M9RPLX2oaJWYFaWHc645KJzsZa/view?usp=sharing).


## Run baseline and our method
To run the baseline code please use the following scripts:

<code> scripts_gqe\train_gqe_DB15k_baseline.sh</code>

<code> scripts_q2b\train_q2b_DB15k_baseline.sh </code>

<code> scripts_q2p\train_q2p_DB15k_baseline.sh </code>

To run the NRN model with the sinusodal numerical encoder:

<code> scripts_gqe\train_gqe_DB15k_value_typed_position.sh </code>

<code> scripts_q2b\train_q2b_DB15k_value_typed_position.sh </code>

<code> scripts_q2p\train_q2p_DB15k_value_typed_position.sh </code>


To run the NRN model with the DICE numerical encoder:

<code> scripts_gqe\train_gqe_DB15k_value_typed_dice.sh </code>

<code> scripts_q2b\train_q2b_DB15k_value_typed_dice.sh </code>

<code> scripts_q2p\train_q2p_DB15k_value_typed_dice.sh </code>


During the running process, you can monitor the training process via tensorboard. The default log storage will be <code> logs/gradient_tape </code>. 

If you have any questions, please contact me vis <code>jbai@connect.ust.hk</code>.
If you find the paper and code useful, please cite our paper. 

<code>@inproceedings{DBLP:conf/kdd/BaiLLYYS23,
  author       = {Jiaxin Bai and
                  Chen Luo and
                  Zheng Li and
                  Qingyu Yin and
                  Bing Yin and
                  Yangqiu Song},
  editor       = {Ambuj K. Singh and
                  Yizhou Sun and
                  Leman Akoglu and
                  Dimitrios Gunopulos and
                  Xifeng Yan and
                  Ravi Kumar and
                  Fatma Ozcan and
                  Jieping Ye},
  title        = {Knowledge Graph Reasoning over Entities and Numerical Values},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {57--68},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3580305.3599399},
  doi          = {10.1145/3580305.3599399},
  timestamp    = {Mon, 25 Sep 2023 08:29:22 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/BaiLLYYS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}   
} </code>








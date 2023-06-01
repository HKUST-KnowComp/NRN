# NRN

The official implementation of the paper: Knowledge Graph Reasoning over Entities and Numerical Values

The input files for the models can be downloaded here. After decompressing, they should be put in the root directory of this repository. The are two directories in the zip file, and the small one serves for debuging and unit testing. They can be downloaded from 
[here](https://drive.google.com/file/d/1QTX1i5M9RPLX2oaJWYFaWHc645KJzsZa/view?usp=sharing).

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



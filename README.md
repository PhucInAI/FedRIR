# Federated Feature Unlearning

[ArXiv](https://arxiv.org/abs/2405.17462) 

### Official pytorch implementation of the paper:

- **Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity**


## Description

The advent of Federated Learning (FL) highlights the practical necessity for the 'right to be forgotten' for all clients, allowing them to request data deletion from the machine learning model's service provider. This necessity has spurred a growing demand for Federated Unlearning (FU). Feature unlearning has gained considerable attention due to its applications in unlearning sensitive features, backdoor features, and bias features. 

Existing methods employ the influence function to achieve feature unlearning, which is impractical for FL as it necessitates the participation of other clients in the unlearning process. Furthermore, current research lacks an evaluation of the effectiveness of feature unlearning. 

<p align="center"> <img src="images/method.png" alt="Methodology" style="zoom: 100%" />
<p align="center"> Figure 1: Overview of our proposed Federated Feature Unlearning framework. </p>

To address these limitations, we define feature sensitivity in the evaluation of feature unlearning according to Lipschitz continuity. This metric characterizes the rate of change or sensitivity of the model output to perturbations in the input feature. We then propose an effective federated feature unlearning framework called Ferrari, which minimizes feature sensitivity. Extensive experimental results and theoretical analysis demonstrate the effectiveness of Ferrari across various feature unlearning scenarios, including sensitive, backdoor, and biased features.

<p align="center"> <img src="images/feature_sensivity.gif" alt="Feature Sensitivity" style="zoom: 50%" />
<p align="center"> Figure 2: Illustration demonstrating the optimization of feature sensitivity for achieving feature unlearning. </p>

## Getting started

### Preparation

Before executing the project code, please prepare the Python environment according to the `requirement.txt` file. We set up the environment with `python 3.9.12` and `torch 2.0.0`. 

```python
pip install -r requirement.txt
```

### How to run

**1. Federated Model Training**

Default ResNet-18 model for image datasets and a fully-connected neural network linear model for tabular datasets.

```python
python fl_training_main.py -train_method baseline -train_mode backdoor -dataset Cifar10 \ 
-global_epochs 100 -local_epochs 10 -batch_size 128 -lr 0.0001  -client_num 10 -frac 0.2 -client_perc 0.1 -save_model True  
```

**2. Federated Feature Unlearning**

```python
python unlearn_main.py -unlearning_scenario backdoor -client_perc 0.1 -dataset Cifar10 \ 
-sample_number 20 -min_sigma 0.05 -max_sigma 1.0 -batch_size 128 -lr 0.0001 -save_model True
```

## Citation
If you find this work useful for your research, please cite
```bibtex
@article{ferrari,
      title={Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity}, 
      author={Hanlin Gu and WinKent Ong and Chee Seng Chan and Lixin Fan},
      year={2024},
      eprint={2405.17462},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.17462}, 
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the author by sending an email to
`winkent.ong@um.edu.my`.

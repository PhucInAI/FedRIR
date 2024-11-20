# Federated Feature Unlearning

[ArXiv](https://arxiv.org/abs/2405.17462) | Supplementary Material | Poster

### Official pytorch implementation of the paper: "Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity"

#### NeurIPS 2024

[![Paper](https://img.shields.io/badge/cs.CV-arxiv:2312.05849-B31B1B.svg)]((https://arxiv.org/abs/2405.17462))
[![Page Views Count](https://badges.toozhao.com/badges/01J9P2R033FHASG7RVP6QSTE3P/green.svg)](https://badges.toozhao.com/stats/01J9P2R033FHASG7RVP6QSTE3P "Get your own page views count badge on badges.toozhao.com")

#### (Released on October 08, 2024)

## Introduction

The advent of Federated Learning (FL) highlights the practical necessity for the 'right to be forgotten' for all clients, allowing them to request data deletion from the machine learning model's service provider. This necessity has spurred a growing demand for Federated Unlearning (FU). Feature unlearning has gained considerable attention due to its applications in unlearning sensitive features, backdoor features, and bias features. 

Existing methods employ the influence function to achieve feature unlearning, which is impractical for FL as it necessitates the participation of other clients in the unlearning process. Furthermore, current research lacks an evaluation of the effectiveness of feature unlearning. 

<p align="center"> <img src="images/method.png" alt="Methodology" style="zoom: 100%" />
<p align="center"> Figure 1: Overview of our proposed Federated Feature Unlearning framework. </p>

To address these limitations, we define feature sensitivity in the evaluation of feature unlearning according to Lipschitz continuity. This metric characterizes the rate of change or sensitivity of the model output to perturbations in the input feature. We then propose an effective federated feature unlearning framework called Ferrari, which minimizes feature sensitivity. Extensive experimental results and theoretical analysis demonstrate the effectiveness of Ferrari across various feature unlearning scenarios, including sensitive, backdoor, and biased features.

<p align="center"> <img src="images/feature_sensivity.gif" alt="Feature Sensitivity" style="zoom: 50%" />
<p align="center"> Figure 2: Illustration demonstrating the optimization of feature sensitivity for achieving feature unlearning. </p>

## Citation
If you find this work useful for your research, please cite
```bibtex
@inproceedings{ferrari,
               title={Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity}, 
               author={Hanlin Gu and WinKent Ong and Chee Seng Chan and Lixin Fan},
               journal={Advances in Neural Information Processing Systems},
               year={2024},
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the author by sending an email to
`winkent.ong@um.edu.my` or `cs.chan@um.edu.my`

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2024 Universiti Malaya.

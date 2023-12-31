[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Robotic Transformer 2 (RT-2): The Vision-Language-Action Model Meets Hardware Abstraction Layer
![rt gif](rt.gif)

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/stargazers) 
[![GitHub license](https://img.shields.io/github/license/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/blob/master/LICENSE)



</div>

---


Robotic Transformer 2 (RT-2) leverages both web and robotics data to generate actionable instructions for robotic control. 

[CLICK HERE FOR THE PAPER](https://robotics-transformer2.github.io/assets/rt2.pdf)


## Installation

RT-2 can be easily installed using pip:

```bash
pip install rt2
```

Additionally, you can manually install the dependencies:

```bash
pip install -r requirements.txt
```

# Usage


The `RT2` class is a PyTorch module that integrates the PALM-E model into the RT-2 class. Here are some examples of how to use it:

#### Initialization

First, you need to initialize the `RT2` class. You can do this by providing the necessary parameters to the constructor:

```python

import torch 
from rt2.model import RT2

model = RT2()

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

# compute the train logits
train_logits = model.train(video, instructions)

# set the model to evaluation mode
model.model.eval()

# compute the eval logits with a conditional scale of 3
eval_logits = model.eval(video, instructions, cond_scale=3.)

```


## Benefits

RT-2 stands at the intersection of vision, language, and action, delivering unmatched capabilities and significant benefits for the world of robotics.

- Leveraging web-scale datasets and firsthand robotic data, RT-2 provides exceptional performance in understanding and translating visual and semantic cues into robotic control actions.
- RT-2's architecture is based on well-established models, offering a high chance of success in diverse applications.
- With clear installation instructions and well-documented examples, you can integrate RT-2 into your systems quickly.
- RT-2 simplifies the complexities of multi-domaster understanding, reducing the burden on your data processing and action prediction pipeline.

## Model Architecture

RT-2 integrates a high-capacity Vision-Language model (VLM), initially pre-trained on web-scale data, with robotics data from RT-2. The VLM uses images as input to generate a sequence of tokens representing natural language text. To adapt this for robotic control, RT-2 outputs actions represented as tokens in the model’s output.

RT-2 is fine-tuned using both web and robotics data. The resultant model interprets robot camera images and predicts direct actions for the robot to execute. In essence, it converts visual and language patterns into action-oriented instructions, a remarkable feat in the field of robotic control.

# Datasets
| Dataset | Description | Source | Percentage in Training Mixture (RT-2-PaLI-X) | Percentage in Training Mixture (RT-2-PaLM-E) |
|---------|-------------|--------|----------------------------------------------|----------------------------------------------|
| WebLI | Around 10B image-text pairs across 109 languages, filtered to the top 10% scoring cross-modal similarity examples to give 1B training examples. | Chen et al. (2023b), Driess et al. (2023) | N/A | N/A |
| Episodic WebLI | Not used in co-fine-tuning RT-2-PaLI-X. | Chen et al. (2023a) | N/A | N/A |
| Robotics Dataset | Demonstration episodes collected with a mobile manipulation robot. Each demonstration is annotated with a natural language instruction from one of seven skills. | Brohan et al. (2022) | 50% | 66% |
| Language-Table | Used for training on several prediction tasks. | Lynch et al. (2022) | N/A | N/A |



# Appreciation

* Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski,
* Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
* Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog,
* Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang,
* Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch,
* Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
* Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong,
* Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu,
* and Brianna Zitkovich

for writing this amazing paper and advancing Humanity

* LucidRains for providing the base repositories for [PALM](https://github.com/lucidrains/PaLM-rlhf-pytorch) and [RT-1](https://github.com/kyegomez/RT-2)

* Any you yes the Human looking at this right now, I appreciate you and love you.

## Commercial Use Cases

The unique capabilities of RT-2 open up numerous commercial applications:

- **Automated Factories**: RT-2 can significantly enhance automation in factories by understanding and responding to complex visual and language cues.
- **Healthcare**: In robotic surgeries or patient care, RT-2 can assist in understanding and performing tasks based on both visual and verbal instructions.
- **Smart Homes**: Integration of RT-2 in smart home systems can lead to improved automation, understanding homeowner instructions in a much more nuanced manner.

## Examples and Documentation

Detailed examples and comprehensive documentation for using RT-2 can be found in the [examples](https://github.com/kyegomez/RT-2/tree/master/examples) directory and the [documentation](https://github.com/kyegomez/RT-2/tree/master/docs) directory, respectively.

## Contributing

Contributions to RT-2 are always welcome! Feel free to open an issue or pull request on the GitHub repository.

## License

RT-2 is provided under the MIT License. See the LICENSE file for details.

## Contact

For any queries or issues, kindly open a GitHub issue or get in touch with [kyegomez](https://github.com/kyegomez).

## Citation

```
@inproceedings{RT-2,2023,
  title={},
  author={Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski,
Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog,
Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang,
Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch,
Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong,
Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu,
and Brianna Zitkovich},
  year={2024}
}
```
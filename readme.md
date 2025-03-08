# Deep Learning Enabled Semantic Communication Systems

Original author: <center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of semantic communication systems based on DeepSC model, with Flask as front-end.

## Acknowledgements
A special thank you to the original authors of the open-source code framework used in this project. Your work has been instrumental in helping me develop my own implementation. I have made modifications to suit my specific needs, but your foundational work is greatly appreciated.

## Requirements
+ See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.

## Bibtex--The original essay information
```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}
```
## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz  
tar zxvf europarl.tgz
python preprocess_text.py
```
## Notes
I have used other dataset to train this model, from wider and newer text dataset.

## Train
```shell
python main.py 
```
### Notes
+ Please carefully set the $\lambda$ of mutual information part since I have tested the model in different platform, 
i.e., Tensorflow and Pytorch, same $\lambda$ shows different performance.  

## Evaluation
```shell
python performance.py
```
## System display
```shell
python run.py
```

### Notes
+ If you want to compute the sentence similarity, please download the bert model. And the system is based on Flask, remember to install all the mudules in the requirements.txt 
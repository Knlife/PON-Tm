The Chinese version of README.md is [here](README-Chinese.md).  
## How to use PON-Tm
You can use this service directly at https://www.yanglab-mi.org.cn/Pon-Tm/.  
Alternatively, you can use the Docker version of this service, which uses Uniref50 as the
background database for calculating PSSM and a larger PLM for processing protein sequences.
This is also the service version used to obtain the results in the paper. You can follow the 
following process to deploy the docker service. 
### 1. pull the image from dockerhub
```shell
docker pull jiejueerqun/pontm
```
### 2.run the image
```shell
docker run -tid -p8088:8000 --name PON-Tm --privileged=true jiejueerqun/pontm /usr/sbin/init
```
### 3.[Get the service in your browser](0.0.0.0:8088)
## Further problems
Contact me at **pluskuang@163.com**.

***
**If you use the code in this repository, please cite.**  
Kuang, J., Zhao, Z., Yang, Y., & Yan, W. (2024). PON-Tm: A Sequence-Based Method for Prediction of
Missense Mutation Effects on Protein Thermal Stability Changes. International journal of molecular
sciences, 25(15), 8379. https://doi.org/10.3390/ijms25158379  
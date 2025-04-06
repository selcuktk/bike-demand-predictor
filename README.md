![bike_demand](https://github.com/user-attachments/assets/fcbcd2a6-d6f5-434d-bd55-8ac6c8140759)

# Bike Demand Predictor Using TensorFlow

Bike Demand Predictor is a deep learning project designed to estimate number of bikes would be rented under given conditions at specific time. **This project was developed to demonstrate the programmer's knowledge of the machine learning projects process and how they apply it.** Additionally, at the end of the project, the created classifiers are evaluated.










## Features

- **No Backend/Frontend:** A lightweight solution focused solely on the deep learning model.


## Installation

- Programming language used and its version: Python 3.8.0

- Install Bike Demand Predictor with Git and Python

```bash
git clone https://github.com/selcuktk/bike-demand-predictor.git
cd bike-demand-predictor
```
- Install required libraries:
```bash
pip install absl-py==2.2.1 astunparse==1.6.3 cachetools==5.5.2 certifi==2025.1.31 charset-normalizer==3.4.1 flatbuffers==2.0.7 gast==0.4.0 google-auth==2.38.0 google-auth-oauthlib==0.4.6 google-pasta==0.2.0 grpcio==1.70.0 h5py==3.11.0 idna==3.10 importlib_metadata==8.5.0 joblib==1.4.2 keras==2.7.0 Keras-Preprocessing==1.1.2 libclang==18.1.1 Markdown==3.7 MarkupSafe==2.1.5 numpy==1.24.4 oauthlib==3.2.2 opt_einsum==3.4.0 pip==25.0.1 protobuf==3.19.0 pyasn1==0.6.1 pyasn1_modules==0.4.2 requests==2.32.3 requests-oauthlib==2.0.0 rsa==4.9 scikit-learn==1.3.2 scipy==1.10.1 setuptools==41.2.0 six==1.17.0 tensorboard==2.7.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1 tensorflow==2.7.0 tensorflow-estimator==2.7.0 tensorflow-io-gcs-filesystem==0.31.0 termcolor==2.4.0 threadpoolctl==3.5.0 typing_extensions==4.13.0 urllib3==2.2.3 Werkzeug==3.0.6 wheel==0.45.1 wrapt==1.17.2 zipp==3.20.2
```
- Version of the used libraries
```bash
Package                      Version
---------------------------- ---------
absl-py                      2.2.1
astunparse                   1.6.3
cachetools                   5.5.2
certifi                      2025.1.31
charset-normalizer           3.4.1
flatbuffers                  2.0.7
gast                         0.4.0
google-auth                  2.38.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.70.0
h5py                         3.11.0
idna                         3.10
importlib_metadata           8.5.0
joblib                       1.4.2
keras                        2.7.0
Keras-Preprocessing          1.1.2
libclang                     18.1.1
Markdown                     3.7
MarkupSafe                   2.1.5
numpy                        1.24.4
oauthlib                     3.2.2
opt_einsum                   3.4.0
pip                          25.0.1
protobuf                     3.19.0
pyasn1                       0.6.1
pyasn1_modules               0.4.2
requests                     2.32.3
requests-oauthlib            2.0.0
rsa                          4.9
scikit-learn                 1.3.2
scipy                        1.10.1
setuptools                   41.2.0
six                          1.17.0
tensorboard                  2.7.0
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.7.0
tensorflow-estimator         2.7.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    2.4.0
threadpoolctl                3.5.0
typing_extensions            4.13.0
urllib3                      2.2.3
Werkzeug                     3.0.6
wheel                        0.45.1
wrapt                        1.17.2
zipp                         3.20.2
```

## The Thought Process Behind the Project

This approach is based on two fundamental principles: a structured recipe and orthogonalization, ensuring clarity and efficiency throughout the implementation.

- [Basic Recipe for Machine Learning](https://www.youtube.com/watch?v=C1N_PDHuJ6Q)
![couırse2 video3 basic recipe](https://github.com/user-attachments/assets/dd0b4bb4-be9d-48df-b66a-990219e2188f)

- [Orthogonalization Principle](https://www.youtube.com/watch?v=UEtvV1D6B3s&t=35s)
![orthogonalization-notes](https://github.com/user-attachments/assets/56ee576e-99b5-41bf-a740-a87fcf4a2262)

Firstly, one starter model is implemented and the path on the recipe is followed considering orthogonalization logic in the notes. After creating the starter classifier, considering parameters, different versions of it, are created.
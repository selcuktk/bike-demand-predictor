![bike_demand](https://github.com/user-attachments/assets/fcbcd2a6-d6f5-434d-bd55-8ac6c8140759)

# Bike Demand Predictor Using TensorFlow

Bike Demand Predictor is a deep learning project designed to estimate number of bikes would be rented under given conditions at specific time. **This project was developed to demonstrate the programmer's knowledge of the machine learning projects process and how they apply it.** Additionally, at the end of the project, the created classifiers are evaluated.


## Features

- **No Backend/Frontend:** A lightweight solution focused solely on the deep learning model.


## Installation

- Programming language used and its version: Python 3.8.0  (pip 25.0.1 used)

- Install Bike Demand Predictor with Git and Python

```bash
git clone https://github.com/selcuktk/bike-demand-predictor.git
cd bike-demand-predictor
```
- Install required libraries:
```bash
pip install absl-py==2.2.1 astunparse==1.6.3 cachetools==5.5.2 certifi==2025.1.31 charset-normalizer==3.4.1 cycler==0.12.1 flatbuffers==2.0.7 fonttools==4.57.0 gast==0.4.0 google-auth==2.38.0 google-auth-oauthlib==0.4.6 google-pasta==0.2.0 grpcio==1.70.0 h5py==3.11.0 idna==3.10 importlib_metadata==8.5.0 joblib==1.4.2 keras==2.7.0 Keras-Preprocessing==1.1.2 kiwisolver==1.4.7 libclang==18.1.1 Markdown==3.7 MarkupSafe==2.1.5 matplotlib==3.5.3 numpy==1.24.4 oauthlib==3.2.2 opt_einsum==3.4.0 packaging==24.2 pandas==2.0.3 pillow==10.4.0 protobuf==3.19.0 pyasn1==0.6.1 pyasn1_modules==0.4.2 pyparsing==3.1.4 python-dateutil==2.9.0.post0 pytz==2025.2 requests==2.32.3 requests-oauthlib==2.0.0 rsa==4.9 scikit-learn==1.3.2 scipy==1.10.1 six==1.17.0 tensorboard==2.7.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1 tensorflow==2.7.0 tensorflow-estimator==2.7.0 tensorflow-io-gcs-filesystem==0.31.0 termcolor==2.4.0 threadpoolctl==3.5.0 typing_extensions==4.13.0 tzdata==2025.2 ucimlrepo==0.0.7 urllib3==2.2.3 Werkzeug==3.0.6 wheel==0.45.1 wrapt==1.17.2 zipp==3.20.2
```
- Version of the used libraries
```bash
Package                      Version
---------------------------- -----------
absl-py                      2.2.1
astunparse                   1.6.3
cachetools                   5.5.2
certifi                      2025.1.31
charset-normalizer           3.4.1
cycler                       0.12.1
flatbuffers                  2.0.7
fonttools                    4.57.0
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
kiwisolver                   1.4.7
libclang                     18.1.1
Markdown                     3.7
MarkupSafe                   2.1.5
matplotlib                   3.5.3
numpy                        1.24.4
oauthlib                     3.2.2
opt_einsum                   3.4.0
packaging                    24.2
pandas                       2.0.3
pillow                       10.4.0
pip                          25.0.1
protobuf                     3.19.0
pyasn1                       0.6.1
pyasn1_modules               0.4.2
pyparsing                    3.1.4
python-dateutil              2.9.0.post0
pytz                         2025.2
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
tzdata                       2025.2
ucimlrepo                    0.0.7
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
![orthogonalization-notes](https://github.com/user-attachments/assets/55c4fda9-bfb1-47f7-9e83-a1cde3db82ad)

Firstly, one starter model is implemented and the path on the recipe is followed considering orthogonalization logic in the notes. After creating the starter classifier, considering parameters, different versions of it, are created.

1. Features of the starter model:

| Feature              | Details                          |
|---------------------|---------------------------------|
| Architecture        | Fully Connected (Dense) Layers  |
| Layers (Neurons)    | 512 → 256 → 32 → 8 → 1 (Output) |
| Activation Function | ReLU (All layers except output)  |
| Output Activation   | None (Linear Output)             |
| Optimizer           | Adam                             |
| Learning Rate       | 0.0005                           |
| Loss Function       | Mean Squared Error (MSE)         |
| Evaluation Metric   | Mean Absolute Error (MAE)        |
| Mini-batch Size     | 32                               |

2. The following results are for the starter model:

![stats](https://github.com/user-attachments/assets/0f137007-7a29-4179-9bff-66a7c9ea4ede)
![stats2](https://github.com/user-attachments/assets/288b2e84-cbc0-431e-ad46-3e84928d49c8)

- Bayes error is unknown because of some factors. Nature of data, it is collected from real world and this collected data can be affected from some random events and it causes noise. Moreover, even with the perfect features human decisions to rent bikes are not deterministic. Two identical days may yield different rental counts. The goal is to reduce the avoidable error (bias + variance), but accept the existence of irreducible noise.

- On the other hand, it can be said that the starter model is fitted well on training set (MAE ~13 is pretty small considering the average target is ~191). However there is variance problem. Because mean absolute error of test data is 116% more than mean absolute error of train data.

3. According to the mentioned recipe, there are 3 options such that more data, regularization or different NN architectures. On this step regularization will be applied on the model.

![stats3](https://github.com/user-attachments/assets/a8d71543-b25c-46aa-b27e-9429c310dbcf)
![stats4](https://github.com/user-attachments/assets/e108af80-ff9e-4971-bcb6-cb649d42a9cd)


After regularization steps, it can be fairly said that high variance situation is fixed, mean squared error value of test is higher compared to train data. It means predictions of train is more stable compared to test. Having higher standard deviation on the errors is normal for a test data compared to training data.

4. [Tuning Process](https://youtu.be/AXDByU3D1hA?si=CZ0ooK_WZxECV-Lo)
![tuning](https://github.com/user-attachments/assets/7c0a323c-cef8-4396-8cbd-e7ab90490a11)
- Considering the source above, following 6 new model are created.
![models](https://github.com/user-attachments/assets/eb8f67f2-728b-4501-a01e-f8c68281064b)

5. [Output](https://github.com/user-attachments/assets/d6082705-6718-46a0-bfb4-db19ee6be04b) and Error Graphs
![mae_graph](https://github.com/user-attachments/assets/a892a812-842a-4a18-a14b-0705e050258e)
![mse graph](https://github.com/user-attachments/assets/2f0424a7-0f72-45b7-bf40-b28b01416b0d)

6. Model Selection Based on Evaluation Metrics

![stats_models](https://github.com/user-attachments/assets/cd24221c-3b10-45e6-9e73-bff387c0d4f9)
- Final results on the models can be seen above. Now, an evaluation metric is going to improved. 
- Humans are bad at estimating this numarical data compared to image classification tasks. Therefore try to decide Bayes error considering human level performance is not useful in this project. Also there is no certain way to decide Bayes error such deciding by considering similar projects. So we are going to decide a goal and put that goal into a metric. 
- There are 3 priorities for the metric (in order of importance): 
    1. Test MAE (accuracy on unseen data)
    2. MAE Variance (stability/generalization)
    3. Test MSE (for avoiding outliers)
- Evaluation metric can be a Composite_Score = α * (Test MAE) + β * (Variance %) + γ * (Test MSE / 100) where α=0.6,  β=0.3, γ=0.1
- Following graph represents composite scores of the models
![composite scores](https://github.com/user-attachments/assets/2949a507-84cd-4eb4-a2b1-5dd0fe7548c0)

As can be seen in the graph, model 6 is the best predictor compared to other predictor models. Considering a small value between scores of model 5 and model 6, evaluation metric can be updated as customers needs after more detailed analysis.
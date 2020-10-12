# Covid19_hotspot_prediction
<img src="https://user-images.githubusercontent.com/40876087/95192127-b0933800-080c-11eb-9119-925df2090b39.JPG" width="90%"></img>

This repository is the implementation of "Covid19_hotspot_prediction" project.(KT-KAIST AI One Team Cooperation)<br>
There are two types of models, Covid-19 and Flu, and the difference between the models is the predicted time interval and the data used.

For the data, I used the information on the movements of the confirmed patient and the data on the floating population, and temporal signal data. 
The amount of movement was reconstructed into inter-/intra-flow between administrative districts to make graph structure data. 
Unfortunately, however, data cannot be disclosed due to KT's personal information policy.<br>

The model consists of: Insert the sequence of graph data into the Graph Convolution Network(GCN) and extract a feature containing the spatial information. 
Then, the LSTM fed the feature sequence predicts future regional risks index.

My implementations used [Pytorch-geometric] for GCN.(https://github.com/rusty1s/pytorch_geometric)

## Requirements

I run the code in the following environment using Anaconda.

- Python >= 3.5
- Pytorch == 1.4
- torchvision == 0.5
- Pytorch-geometric >= 1.5.0

## Training

If you want to train Covid-19 model, run this command:

```train
/covid_models/run_covid.sh
```

If you want to train Flu model, run this command:

```train
/flu_models/run_flu.sh
```

If you want to see and change the arguments of training code, run this command:
```
python3 covid_models/main.py --help
```

## Evaluation

To evaluate the model(s) and see the results, please refer to the `Evaluation.ipynb`

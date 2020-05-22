## Build a movie recommender using matrix factorization with MLNet.AutoPipeline
This example shows how to use MLNet.AutoPipeline to build and optimize a ML.NET pipeline.

The corresponding tutorial on ML.NET is here: [Build a movie recommender using matrix factorization with ML.NET](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/movie-recommendation), it uses the same training/testing dataset and pipeline, except it uses default parameter setting for matrix factorization trainer. By comparing the two experiment result, we can find out how much improvement it can gain after parameter optimization.

## Dataset
The dataset consists of around 1 million rows of rating between users and movies. Below is a preview of data.

| userId | movieId | rating | timestamp |
|-|-|-|-|
|1|1|4|964982703|
|1|3|4|964981247|
|1|6|4|964982224|

In the dataset, there're four columns:
- `userId`: user id.
- `movieId`: movie id.
- `rating`: rating (max 5).
- `timestamp`: time stamp.

Train dataset: `recommendation-ratings-train.csv`
Test dataset: `recommendation-ratings-test.csv`

## Pipeline
`ValueToKeyMappingEstimator=>ValueToKeyMappingEstimator=>MatrixFactorizationTrainer=>ColumnCopyingEstimator`

## Sweeping range for matrix factorization trainer's hyper-parameter.
- `NumberOfIterations`: [10, 100]
- `C`: [1e-5, 0.1]
- `Alpha`: [1e-4, 1]
- `ApproximationRank`: [128, 512]
- `Lambda`: [0.01, 10]
- `LearningRate`: [1e-3, 0.1] 

## Experiment result
||RMSE|
|-|-|
|without parameter optimization|[`0.9940`](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/movie-recommendation#evaluate-your-model)|
|with parameter optimization| `0.8972`|


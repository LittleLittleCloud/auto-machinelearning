## Automate ML for [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

- MLNet.AutoPipeline: build and optimize [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) pipelines over pre-defined hyper parameters.

- MLNet.Expert: Automated machine learning based on [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) (about-to-come).

[![Build Status](https://dev.azure.com/xiaoyuz0315/BigMiao/_apis/build/status/LittleLittleCloud.machinelearning-auto-pipeline?branchName=master)](https://dev.azure.com/xiaoyuz0315/BigMiao/_build/latest?definitionId=1&branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

First, add `MLNet.AutoPipeline` and `MLNet.Expert` to your project. You can get those packages from our [nightly build](#Installation).

```xml
<ItemGroup>
  <PackageReference Include="MLNet.AutoPipeline" />
  <PackageReference Include="MLNet.Expert" />
</ItemGroup>
```
Then create an auto-pipeline with `NumericFeatureExpert` and `ClassificationExpert`. The following code creates an auto-pipeline on `Iris` dataset that can achieve 100% accuracy on its test dataset.
```csharp
var context = new MLContext();
var normalizeExpert = new NumericFeatureExpert(context, new NumericFeatureExpert.Option());
var classificationExpert = new ClassificationExpert(context, new ClassificaitonExpert.Option());

var estimatorChain = context.Transforms.Conversion.MapValueToKey("species", "species")
                    .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                    // sweep over NormalizeMeanVariance, NormalizeMinMax or No-op transformer
                    .Append((normalizeExpert.Propose("features") as EstimatorNodeGroup).OrNone())
                    // sweep over NaiveBayes, LbfgsMaximumEntropy, LightGBM, SdcaMaximumEntropy, FastTreeOva and FastForestOva
                    .Append(classificationExpert.Propose("species", "features"));
```

Then create an `Experiment` to sweep over `estimatorChain` to find the best pipeline and hyperparameter.

```csharp
var experimentOption = new Experiment.Option()
{
    ScoreMetric = new MicroAccuracyMetric(), // Use Micro Accuracy as score.
    Sweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option()), // Use GaussProcess sweeper to optimize hyperparameters.
    Iteration = 30, // try 30 times for each pipeline
    Label = "species", // dataset label name
};

var experiment = new Experiment(context, estimatorChain, experimentOption);
var result = await experiment.TrainAsync(split.TrainSet, split.TestSet); // train experiment.
```

The training result is saved in `result`, which includes best model, best hyperparameter and other necessary informations.
```csharp
var bestModel = result.BestModel;
context.Model.Save(bestModel, dataset.Schema, "bestmodel.zip");
Console.WriteLine($"best score: {result.BestIteration.ScoreMetric.Score}");
Console.WriteLine($"training time: {result.TrainingTime}");
```


## Examples
- Iris classification
- Sentiment analyasis
- [Movie recommendation](examples/Movie&#32;Recommendation/README.md)

## Installation

This project is still under developing, so no released package is available yet. However, you can get the latest build via our nightly feed by adding this URL to your NuGet.Config

`https://pkgs.dev.azure.com/xiaoyuz0315/BigMiao/_packaging/MLNet-Auto-Pipeline%40Local/nuget/v3/index.json`

## Contributing
We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md)

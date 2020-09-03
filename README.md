## **MLNet.AutoPipeline**: AutoML for [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

**ML.Net AutoPipeline** is a set of packages build on top of ML.Net that provide AutoML feature. It is aimed to solve the two following problems that vastly exists in Machinelearning:
- Given a ML pipeline, find the best hyper-parameters for its transformers or trainers.
- Given a dataset and a ML task, find the best pipeline for solving this task.

[![Build Status](https://dev.azure.com/xiaoyuz0315/BigMiao/_apis/build/status/LittleLittleCloud.auto-machinelearning?branchName=master)](https://dev.azure.com/xiaoyuz0315/BigMiao/_build/latest?definitionId=3&branchName=master) ![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/xiaoyuz0315/BigMiao/3?color=green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Try it on Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LittleLittleCloud/MLNet-AutoPipeline-Examples/master)

## Quick Start

First, add `MLNet.AutoPipeline` to your project. You can get those packages from our [nightly build](#Installation).

```xml
<ItemGroup>
  <PackageReference Include="MLNet.AutoPipeline" />
</ItemGroup>
```
Then create a `SweepablePipeline` using `AutoPipelineCatalog` API. `SweepablePipeline` is similar to the concept of [`EstimatorChain`](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.estimatorchain-1?view=ml-dotnet) in ML.Net. And it will fine-tune hyperparameters by sweeping over a group of pre-defined parameters during training.

```csharp
var context = new MLContext();
var sweepablePipeline = context.Transforms.Conversion.MapValueToKey("species", "species")
                    // here we use Iris dataset as example.
                    .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                    // create a sweepable LbfgsMaximumEntropy trainer
                    .Append(context.AutoML().MultiClassification.LbfgsMaximumEntropy("species", "features"));
```

Then create an `Experiment` to sweep over `sweepablePipeline` to find the best pipeline and hyperparameter.

```csharp
var experimentOption = new Experiment.Option()
{
    ScoreMetric = new MicroAccuracyMetric(), // Use Micro Accuracy as score.
    Label = "species", // dataset label name
};

var experiment = context.AutoML().CreateExperiment(estimatorChain, experimentOption)
var result = await experiment.TrainAsync(split.TrainSet); // train experiment.
```

## Examples
Please visit [MLNet-AutoPipeline-Example](https://github.com/LittleLittleCloud/MLNet-AutoPipeline-Examples) for MLNet.AutoPipeline examples.

## Installation

This project is still under developing, so no released package is available yet. However, you can get the latest build via our nightly feed by adding this URL to your NuGet.Config

`https://pkgs.dev.azure.com/xiaoyuz0315/BigMiao/_packaging/MLNet-Auto-Pipeline%40Local/nuget/v3/index.json`

## Contributing
We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md)

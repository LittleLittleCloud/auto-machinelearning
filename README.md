## **MLNet.AutoPipeline**: AutoML for [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

**ML.Net AutoPipeline** is a set of packages build on top of ML.Net that provide AutoML feature. In short, it is aimed to solve the two following problems that vastly exists in Machinelearning:
- Given a ML pipeline, find the best hyper-parameters for its transformers or trainers.
- Given a dataset and a ML task, find the best pipeline for solving this task.

**ML.Net AutoPipeline** solves the first problem by sweepable pipeline, which sweeps over a set of pre-defined hyper-parameters and finds the best candidate. It solves the second problem by `MLNet.Expert`, which automatically constructs pipelines based on dataset and task. 

**ML.Net AutoPipeline** contains four libraries:
-  [MLNet.AutoPipeline](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.AutoPipeline.html): Provides API for creating sweepable MLNet pipelines. 
- [MLNet.Sweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.html): Provides different sweepers which can be used to optimize hyper-parameter in `MLNet.AutoPipeline`. Right now the available sweepers are [UniformRandomSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.UniformRandomSweeper.html), [RandomGridSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.RandomGridSweeper.html) and [GaussProcessSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.GaussProcessSweeper.html).
- [MLNet.Expert](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Expert.html): *(coming soon)* An AutoML library build on top of `MLNet.AutoPipeline`. It's your best choice if you don't want to define pipeline yourself but want to rely on the power of AutoML. As what it name says, it's your MLNet expert.
- [MLNet.CodeGenerator](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.CodeGenerator.html): *(coming soon)* Provides API for generating C# code for creating ML.Net pipeline.

[![Build Status](https://dev.azure.com/xiaoyuz0315/BigMiao/_apis/build/status/LittleLittleCloud.machinelearning-auto-pipeline-pr?branchName=master)](https://dev.azure.com/xiaoyuz0315/BigMiao/_build/latest?definitionId=1&branchName=master) ![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/xiaoyuz0315/BigMiao/1?color=green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
var result = await experiment.TrainAsync(split.TrainSet); // train experiment.
```

The training result is saved in `result`, which includes best model, best hyperparameter and other necessary informations.
```csharp
var bestModel = result.BestModel;
context.Model.Save(bestModel, dataset.Schema, "bestmodel.zip");
Console.WriteLine($"best validate score: {result.BestIteration.ScoreMetric.Score}");
Console.WriteLine($"training time: {result.TrainingTime}");
```


## Examples
Please visit [MLNet-AutoPipeline-Example](https://github.com/LittleLittleCloud/MLNet-AutoPipeline-Examples) for MLNet.AutoPipeline examples.

## Installation

This project is still under developing, so no released package is available yet. However, you can get the latest build via our nightly feed by adding this URL to your NuGet.Config

`https://pkgs.dev.azure.com/xiaoyuz0315/BigMiao/_packaging/MLNet-Auto-Pipeline%40Local/nuget/v3/index.json`

## Contributing
We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md)

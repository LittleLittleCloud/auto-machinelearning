## Automate ML for [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

- MLNet.AutoPipeline: build and optimize [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) pipelines over pre-defined hyper parameters.

- MLNet.Expert: Automated machine learning based on [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) (about-to-come).

[![Build Status](https://dev.azure.com/xiaoyuz0315/BigMiao/_apis/build/status/LittleLittleCloud.machinelearning-auto-pipeline?branchName=master)](https://dev.azure.com/xiaoyuz0315/BigMiao/_build/latest?definitionId=1&branchName=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

With `Expert`, you can easily create an auto-pipeline with a bunch of pre-defined trainers and transformer. The following code shows how to build an auto-pipeline for `Iris` dataset that sweeps over 8 classification trainers and tries [NormalizeMeanVariance](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.normalizationcatalog.normalizemeanvariance?view=ml-dotnet) and [NormalizeMinMax](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.normalizationcatalog.normalizeminmax?view=ml-dotnet) over its numeric features.

```csharp
var context = new MLContext();
var normalizeExpert = new NumericFeatureExpert(context, new NumericFeatureExpert.Option());
var classificationExpert = new ClassificationExpert(context, new ClassificaitonExpert.Option());

var pipelines = context.Transforms.Conversion.MapValueToKey("species", "species")
                    .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                    // sweep over NormalizeMeanVariance, NormalizeMinMax or No-op transformer
                    .Append((normalizeExpert.Propose("features") as EstimatorNodeGroup).OrNone())
                    // sweep over NaiveBayes, LbfgsMaximumEntropy, LightGBM, SdcaMaximumEntropy, FastTreeOva and FastForestOva
                    .Append(classificationExpert.Propose("species", "features"));
```

The `pipelines` will create 24 different pipelines when expanding, and it provides smart sweeper to optimize the best hyperparamater for each pipeline. Here's an example of how to expand `pipelines`.
```csharp

foreach (var sweepablePipeline in pipelines.BuildSweepablePipelines())
{
    // Use Bayesian hyperparameter optimization.
    // Other choices: Random, GridSearch
    var sweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option());
    sweepablePipeline.UseSweeper(sweeper);

    // Sweeping 50 times for each estimator chain to find out best hyperparameter
    foreach (var pipeline in sweepablePipeline.Sweeping(50))
    {
        // Train
        var eval = pipeline.Fit(split.TrainSet).Transform(split.TestSet);
        // Validate
        var metrics = context.MulticlassClassification.Evaluate(eval, "species");
        var result = new RunResult(sweepablePipeline.Sweeper.Current, metrics.MacroAccurac

        // Add the last running result as RunHistory
        // so sweeper can learn from past to produce better hyperparameter for next run.
        sweepablePipeline.Sweeper.AddRunHistory(result);
        Console.WriteLine($"macro accuracy: {metrics.MacroAccuracy}");
    }
}
```


## Examples
- Iris classification
- Sentiment analyasis
- [Movie recommendation](examples/Movie&#32;Recommendation/README.md)

## Installation

This project is still under developing, so no released package is available yet. However, you can get the latest build via our nightly feed by adding this URL to your NuGet.Config

`https://pkgs.dev.azure.com/xiaoyuz0315/BigMiao/_packaging/MLNet-Auto-Pipeline%40Local/nuget/v3/index.json`

## Contributing
We welcome contributions! Please see our [contrubution guide](CONTRIBUTING.md)

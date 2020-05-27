// <copyright file="Program.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Experiment;
using MLNet.AutoPipeline.Extension;
using MLNet.AutoPipeline.Metric;
using MLNet.Expert;
using MLNet.Sweeper;
using System;
using System.Threading.Tasks;

namespace Iris
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            var context = new MLContext();
            var dataset = context.Data.LoadFromTextFile<Iris>(@".\iris.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);
            var normalizeExpert = new NumericFeatureExpert(context, new NumericFeatureExpert.Option());

            var classificationOption = new ClassificationExpert.Option()
            {
                UseNaiveBayes = false,
                UseLbfgsMaximumEntropy = false,
                UseLightGBM = false,
                UseSdcaMaximumEntropy = false,
                UseFastTreeOva = false,
                UseGamOva = false,
            };

            var classificationExpert = new ClassificationExpert(context, classificationOption);

            var estimatorChain = context.Transforms.Conversion.MapValueToKey("species", "species")
                          .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                          .Append((normalizeExpert.Propose("features") as EstimatorNodeGroup).OrNone())
                          .Append(classificationExpert.Propose("species", "features"));

            var experimentOption = new Experiment.Option()
            {
                ScoreMetric = new MicroAccuracyMetric(),
                Sweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option()),
                Iteration = 30,
                Label = "species",
            };

            var experiment = new Experiment(context, estimatorChain, experimentOption);

            var reporter = new Reporter();

            var result = await experiment.TrainAsync(split.TrainSet, split.TestSet, reporter: reporter);

            Console.WriteLine($"best score: {result.BestIteration.ScoreMetric.Score}");
            Console.WriteLine($"training time: {result.TrainingTime}");
        }

        private class Iris
        {
            [LoadColumn(0)]
            public float sepal_length;

            [LoadColumn(1)]
            public float sepal_width;

            [LoadColumn(2)]
            public float petal_length;

            [LoadColumn(3)]
            public float petal_width;

            [LoadColumn(4)]
            public string species;
        }

        private class Reporter : IProgress<IterationInfo>
        {
            public void Report(IterationInfo value)
            {
                Console.WriteLine(value.SweepablePipeline.Summary());
                Console.WriteLine(value.ParameterSet.ToString());
                Console.WriteLine($"validate score: {value.ValidateScoreMetric.Name}: {value.ValidateScoreMetric.Score}");
                Console.WriteLine($"training time: {value.TrainingTime}");
            }
        }
    }
}

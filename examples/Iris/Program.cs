// <copyright file="Program.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Extension;
using MLNet.Expert;
using MLNet.Sweeper;
using System;

namespace Iris
{
    public class Program
    {
        static void Main(string[] args)
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
                UseSdcaNonCalibrated = false,
                UseFastTreeOva = false,
                UseFastForestOva = false,
            };

            var classificationExpert = new ClassificationExpert(context, classificationOption);

            var estimatorChain = context.Transforms.Conversion.MapValueToKey("species", "species")
                          .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                          .Append((normalizeExpert.Propose("features") as EstimatorNodeGroup).OrNone())
                          .Append(classificationExpert.Propose("species", "features"));

            foreach (var sweepablePipeline in estimatorChain.BuildSweepablePipelines())
            {
                Console.WriteLine(sweepablePipeline.Summary());
                var sweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option());
                sweepablePipeline.UseSweeper(sweeper);
                foreach (var pipeline in sweepablePipeline.Sweeping(50))
                {
                    if (sweepablePipeline.Sweeper.Current != null)
                    {
                        Console.WriteLine(sweepablePipeline.Sweeper.Current.ToString());
                    }

                    var eval = pipeline.Fit(split.TrainSet).Transform(split.TestSet);
                    var metrics = context.MulticlassClassification.Evaluate(eval, "species");
                    var result = new RunResult(sweepablePipeline.Sweeper.Current, metrics.MacroAccuracy, true);
                    sweepablePipeline.Sweeper.AddRunHistory(result);
                    Console.WriteLine($"macro accuracy: {metrics.MacroAccuracy}");
                }
            }
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
    }
}

// <copyright file="EstimatorNodeChainE2ETest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.Sweeper;
using MLNet.AutoPipeline.Extension;
using System;
using Xunit;
using Xunit.Abstractions;
using MLNet.Expert;

namespace MLNet.AutoPipeline.Test
{
    public class EstimatorNodeChainE2ETest
    {
        private ITestOutputHelper _output;

        public EstimatorNodeChainE2ETest(ITestOutputHelper output)
        {
            this._output = output;
        }

        [Fact(Skip ="ignore")]
        public void EstiamtorNodeChain_iris_e2eTest_gauss_process_sweeper()
        {
            var context = new MLContext();
            var dataset = context.Data.LoadFromTextFile<Iris>(@".\TestData\iris.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);
            var normalizeExpert = new NumericFeatureExpert(context, new NumericFeatureExpert.Option());
            var classificationExpert = new ClassificationExpert(context, new ClassificationExpert.Option());

            var naiveByaseTrainer = context.MulticlassClassification.Trainers.NaiveBayes("species", "features");

            var lightGMBOption = new LightGBMOption();
            Func<LightGBMOption, LightGbmMulticlassTrainer> lightGBM = (LightGBMOption option) =>
            {
                return context.MulticlassClassification.Trainers.LightGbm("species", "features", learningRate: option.lr, numberOfLeaves: option.leaves, minimumExampleCountPerLeaf: option.countPerLeaf);
            };
            var estimatorChain = context.Transforms.Conversion.MapValueToKey("species", "species")
                          .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                          .Append((normalizeExpert.Propose("features") as EstimatorNodeGroup).OrNone())
                          .Append((classificationExpert.Propose("species", "features") as EstimatorNodeGroup).Append(lightGBM, lightGMBOption));

            foreach ( var sweepablePipeline in estimatorChain.BuildSweepablePipelines())
            {
                this._output.WriteLine(sweepablePipeline.Summary());
                var sweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option());
                sweepablePipeline.UseSweeper(sweeper);
                foreach (var pipeline in sweepablePipeline.Sweeping(200))
                {
                    var eval = pipeline.Fit(split.TrainSet).Transform(split.TestSet);
                    var metrics = context.MulticlassClassification.Evaluate(eval, "species");

                    if (sweepablePipeline.Sweeper.Current != null)
                    {
                        this._output.WriteLine(sweepablePipeline.Sweeper.Current.ToString());
                    }

                    var result = new RunResult(sweepablePipeline.Sweeper.Current, metrics.MacroAccuracy, true);
                    sweepablePipeline.Sweeper.AddRunHistory(result);
                    this._output.WriteLine($"macro accuracy: {metrics.MacroAccuracy}");
                }
            }
        }

        private class LightGBMOption: OptionBuilder<LightGBMOption>
        {
            [Parameter(0.001f ,0.1f ,true,20)]
            public float lr;

            [Parameter(10, 1000, true, 20)]
            public int leaves;

            [Parameter(10, 1000, true, 20)]
            public int iteration;

            [Parameter(10, 1000, true, 20)]
            public int countPerLeaf;
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

// <copyright file="Program.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;
using MLNet.AutoPipeline;
using MLNet.Sweeper;
using System;
using static Microsoft.ML.Trainers.MatrixFactorizationTrainer;

namespace Movie_Recommendation
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var paramaters = new MFOption();
            var train_data = context.Data.LoadFromTextFile<ModelInput>(@".\recommendation-ratings-train.csv", separatorChar: ',', hasHeader: true);
            var test_data = context.Data.LoadFromTextFile<ModelInput>(@".\recommendation-ratings-test.csv", separatorChar: ',', hasHeader: true);

            var gpSweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option() { InitialPopulation = 50 });
            var mfTrainer = new SweepableNode<MatrixFactorizationPredictionTransformer, Options>(context.Recommendation().Trainers.MatrixFactorization, paramaters);

            var pipelines = new SweepablePipeline()
                           .Append(context.Transforms.Conversion.MapValueToKey("userId", "userId"))
                           .Append(context.Transforms.Conversion.MapValueToKey("movieId", "movieId"))
                           .Append(mfTrainer)
                           .Append(context.Transforms.CopyColumns("output", "Score"));

            pipelines.UseSweeper(gpSweeper);
            Console.WriteLine(pipelines.Summary());

            RunResult bestHistory = null;
            foreach (var sweepingInfo in pipelines.Sweeping(100))
            {
                var pipeline = sweepingInfo.Pipeline;
                var eval = pipeline.Fit(train_data).Transform(test_data);
                var metrics = context.Regression.Evaluate(eval, "rating", "Score");
                Console.WriteLine(gpSweeper.Current.ToString());
                var result = new RunResult(gpSweeper.Current, metrics.RootMeanSquaredError, false);

                if (bestHistory == null || bestHistory?.MetricValue > result.MetricValue)
                {
                    bestHistory = result;
                }

                gpSweeper.AddRunHistory(result);
                Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }

            Console.WriteLine($"Best RMSE: {bestHistory.MetricValue}");
        }

        private class MFOption : OptionBuilder<MatrixFactorizationTrainer.Options>
        {
            public string MatrixColumnIndexColumnName = "userId";

            public string MatrixRowIndexColumnName = "movieId";

            public string LabelColumnName = "rating";

            public bool Quiet = true;

            [Parameter(10, 100, false, 20)]
            public int NumberOfIterations = 10;

            [Parameter(0.00001f, 0.1f, true)]
            public float C = 0.00001f;

            [Parameter(0.0001f, 1f, true)]
            public float Alpha = 0.0001f;

            [Parameter(128, 512, steps: 20)]
            public int ApproximationRank = 20;

            [Parameter(0.01f, 10f, true, 100)]
            public double Lambda = 0.01f;

            [Parameter(0.001f, 0.1f, true, 100)]
            public double LearningRate = 0.001f;
        }

        private class ModelInput
        {
            [ColumnName("userId"), LoadColumn(0)]
            public float UserId { get; set; }

            [ColumnName("movieId"), LoadColumn(1)]
            public float MovieId { get; set; }

            [ColumnName("rating"), LoadColumn(2)]
            public float Rating { get; set; }
        }
    }
}

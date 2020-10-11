// <copyright file="Wiki250Benchmark.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using BenchmarkDotNet.Attributes;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.Expert;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Benchmark
{
    public class Wiki250Benchmark : BenchmarkTestBase
    {
        [Benchmark]
        public override void Run()
        {
            var context = new MLContext(1);
            context.Log += this.Context_Log;
            var columns = new List<Column>();
            columns.Add(new Column("Sentiment", ColumnType.Catagorical, ColumnPurpose.Label));
            columns.Add(new Column("SentimentText", ColumnType.String, ColumnPurpose.TextFeature));
            var wiki = this.GetFileFromTestData("wiki.tsv");
            var data = context.Data.LoadFromTextFile<Wiki>(wiki, hasHeader: true);
            var trainTestSplit = context.Data.TrainTestSplit(data);
            var experimentOption = new Experiment.Option()
            {
                EvaluateFunction = (MLContext context, IDataView data) =>
                {
                    return context.BinaryClassification.EvaluateNonCalibrated(data, "Sentiment").Accuracy;
                },
                ParameterSweeperIteration = 5,
            };
            var experiment = context.AutoML().CreateBinaryClassificationExperiment(columns, experimentOption);
            var result = experiment.TrainAsync(trainTestSplit.TrainSet, 0.1f, Reporter.Instance).Result;
            var eval = result.BestModel.Transform(trainTestSplit.TestSet);
            var eval_score = experimentOption.EvaluateFunction(context, eval);
            Console.WriteLine($"eval accuracy: {eval_score}");
        }

        private class Reporter : IProgress<IterationInfo>
        {
            public static Reporter Instance = new Reporter();

            public void Report(IterationInfo value)
            {
                Console.WriteLine(value.EvaluateScore);
            }
        }

        private class Wiki
        {
            [LoadColumn(0)]
            public bool Sentiment { get; set; }

            [LoadColumn(1)]
            public string SentimentText { get; set; }
        }
    }
}

// <copyright file="NniExperimentTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLNet.Expert;
using MLNet.Expert.Serializable;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.NNI.Tests
{
    public class NniExperimentTests : TestBase
    {
        public NniExperimentTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact(Skip ="e2e")]
        public async Task Taxi_fare_price_end_to_end_test_Aysnc()
        {
            var context = new MLContext();
            var input = this.GetFileFromTestData("taxi-fare-train.csv");
            var columnInferenceResult = context.Auto().InferColumns(input, labelColumnName: "fare_amount", separatorChar: ',');
            var dataset = context.Data.LoadFromTextFile(input, columnInferenceResult.TextLoaderOptions);
            var columns = this.CreateColumns(columnInferenceResult.ColumnInformation);
            var pb = new PipelineBuilder(TaskType.Regression, false, true);
            var pipeline = pb.BuildPipeline(context, columns);
            HardCode.BasePath = Path.GetDirectoryName(new Uri(typeof(NniExperimentTests).Assembly.CodeBase).LocalPath);
            var cts = new CancellationTokenSource();
            cts.CancelAfter(60 * 1000);
            var option = new NniExperiment.Option()
            {
                ParameterSweeper = nameof(RandomGridSweeper),
                PipelineSweeperIteration = 2,
                ParameterSweeperIteration = 3,
                PipelineSweeper = nameof(GridSearchSweeper),
                EvaluateFunction = nameof(SerializableEvaluateFunction.RSquare),
                TaskType = TaskType.Regression,
                Label = "fare_amount",
                TrainPath = Path.Combine(HardCode.BasePath, "train.bin"),
                TestPath = Path.Combine(HardCode.BasePath, "test.bin"),
                ModelOutputFolder = Path.Combine(Path.GetTempPath(), "AutoML-NNI"),
            };

            using (var exp = new NniExperiment(pipeline, option))
            {
                exp.OutputHandler += this.Exp_OutputHandler;
                exp.TrialMetricHandler += this.Exp_TrialMetricHandler;
                await exp.StartTrainingAsync(context, dataset, dataset, ct: cts.Token);
            }
        }

        private void Exp_TrialMetricHandler(object sender, TrialMetric e)
        {
            var res = e.GetTrialResult();

            this.Output.WriteLine($"Id: {e.ParameterId}");
            this.Output.WriteLine($"Duriation: {res?.Duration}");
            this.Output.WriteLine($"Metric: {string.Join(",", res?.Metrics.Select(kv => $"{kv.Key}={kv.Value}"))}");
        }

        private void Exp_OutputHandler(object sender, OutputEventArgs e)
        {
            this.Output.WriteLine(e.Data);
        }

        private Column[] CreateColumns(ColumnInformation columns)
        {
            var txtColumns = columns.TextColumnNames.Select(c => new Column(c, ColumnType.String, ColumnPurpose.TextFeature));
            var numericColumns = columns.NumericColumnNames.Select(c => new Column(c, ColumnType.Numeric, ColumnPurpose.NumericFeature));
            var catColumns = columns.CategoricalColumnNames.Select(c => new Column(c, ColumnType.Catagorical, ColumnPurpose.CategoricalFeature));
            var labelColumn = new Column(columns.LabelColumnName, ColumnType.Catagorical, ColumnPurpose.Label);
            return txtColumns.Concat(numericColumns).Concat(catColumns).Append(labelColumn).ToArray();
        }
    }
}

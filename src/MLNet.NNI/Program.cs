using System;
using System.Threading.Tasks;
using Newtonsoft.Json;
using MLNet.Expert;
using System.IO.Pipes;
using System.Text;
using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Sweeper;
using MLNet.Expert.Serializable;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MLNet.NNI;

class Program
{
    public static async Task Main(string[] args)
    {
        if (args.Length == 0) {
            // Configure trial class, tuner, and search space to create experiment
            var context = new MLContext();
            var trainingManagerOption = new NniExperiment.Option()
            {
                ParameterSweeper = nameof(RandomGridSweeper),
                PipelineSweeperIteration = 2,
                ParameterSweeperIteration = 10,
                PipelineSweeper = nameof(GridSearchSweeper),
                EvaluateFunction = nameof(SerializableEvaluateFunction.RSquare),
                TaskType = TaskType.Regression,
                Label = "petal_width",
                TrainPath = Path.Combine(HardCode.BasePath, "train.bin"),
                TestPath = Path.Combine(HardCode.BasePath, "test.bin"),
                ModelOutputFolder = Path.Combine(Path.GetTempPath(), "AutoML-NNI"),
            };

            var pipeline = context.AutoML().Serializable().Transforms.Concatnate(new[] { "sepal_length", "sepal_width", "petal_length" }, "feature")
                                  .Append(
                                    context.AutoML().Serializable().Regression.Gam("petal_width", "feature"),
                                    context.AutoML().Serializable().Regression.LbfgsPoissonRegression("petal_width", "feature"));

            var train = context.Data.LoadFromTextFile<Iris>("iris.csv", hasHeader: true, separatorChar: ',');
            var test = context.Data.LoadFromTextFile<Iris>("iris.csv", hasHeader: true, separatorChar: ',');

            using (var exp = new NniExperiment(pipeline, trainingManagerOption))
            {
                await exp.StartTrainingAsync(context, train, test, Reporter.Instance);
            }

        } else if (args[0] == "--trial") {
            TrialRuntime.Run();

        } else if (args[0] == "--debug") {
            Console.WriteLine("[debug]");
        }

    }

    class Reporter : IProgress<TrialMetric>
    {
        public static Reporter Instance = new Reporter();
        public void Report(TrialMetric value)
        {
            var json = Uri.UnescapeDataString(value.Value);
            var trailResult = JsonConvert.DeserializeObject<TrialResult>(json);
            Console.WriteLine($"validate metric: {string.Join(",", trailResult.Metrics.Select(kv => $"{kv.Key}={kv.Value}"))}");
            Console.WriteLine($"training time: {trailResult.Duration}");
            Console.WriteLine($"model path: {trailResult.ModelPath}");

        }
    }

}

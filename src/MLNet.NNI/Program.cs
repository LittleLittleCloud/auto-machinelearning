using System;
using System.Threading.Tasks;
using Newtonsoft.Json;
using MLNet.Expert;
using System.IO.Pipes;
using System.Text;
using System.Text.Json.Serialization;
using Microsoft.ML;
using MLNet.AutoPipeline;
using nni_lib;
using MLNet.Sweeper;
using MLNet.Expert.Serializable;
using System.Collections.Generic;
using Nni;

class Program
{
    public static async Task Main(string[] args)
    {
        if (args.Length == 0) {
            // Configure trial class, tuner, and search space to create experiment
            var context = new MLContext();
            var trainingManagerOption = new TrainingManager.Option()
            {
                ParameterSweeper = nameof(RandomGridSweeper),
                PipelineSweepingIteration = 2,
                ParameterSweepingIteration = 3,
                PipelineSweeper = nameof(GridSearchSweeper),
                EvaluationMetric = nameof(SerializableEvaluateFunction.RSquare),
                IsAzureAttach = false,
                IsNNITraining = true,
                TaskType = TaskType.Regression,
                Label = "petal_width",
            };

            var pipeline = context.AutoML().Serializable().Transforms.Concatnate(new[] { "sepal_length", "sepal_width", "petal_length" }, "feature")
                                  .Append(
                                    context.AutoML().Serializable().Regression.LightGbm("petal_width", "feature"),
                                    context.AutoML().Serializable().Regression.Sdca("petal_width", "feature"));

            var exp = new Nni.Experiment(context, trainingManagerOption, pipeline, "iris.csv");

            // Select number of trials to run
            int trialNum = trainingManagerOption.PipelineSweepingIteration * trainingManagerOption.ParameterSweepingIteration;
            var result = await exp.Run(trialNum);

            // Print result
            Console.WriteLine("=== Experiment Result ===");
            foreach (var kv in result)
            {
                (string parameter, double metric) = kv;
                Console.WriteLine($"Parameter: {parameter}  Result: {metric}");
            }

        } else if (args[0] == "--trial") {
            Nni.TrialRuntime.Run();

        } else if (args[0] == "--debug") {
            Console.WriteLine("[debug]");
        }

    }

}

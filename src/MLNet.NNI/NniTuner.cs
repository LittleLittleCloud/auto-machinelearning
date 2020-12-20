using MLNet.AutoPipeline;
using MLNet.Expert;
using MLNet.Expert.Contract;
using MLNet.Sweeper;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MLNet.NNI
{
    class NniTuner : ITuner
    {
        private ISweeper pipelineSweeper;
        private ISweeper parameterSweeper;
        private ISweepable<SingleEstimatorSweepablePipeline> pipeline;
        private List<IRunResult> pipelineRunHistory;
        private string trainDataPath = Path.GetFullPath("iris.csv");
        private string testDataPath = Path.GetFullPath("iris.csv");
        private int pipelineSweeperIter;
        private int parameterSweeperIter;
        private string evaludateFunctionName;
        private string label;
        private List<(SingleEstimatorSweepablePipelineDataContract, IDictionary<string, string>)> generateParameters = new List<(SingleEstimatorSweepablePipelineDataContract, IDictionary<string, string>)>();

        public NniTuner(
            ISweepable<SingleEstimatorSweepablePipeline> pipeline,
            ISweeper pipelineSweeper,
            int pipelineSweeperIteration,
            ISweeper parameterSweeper,
            int parameterIteration,
            string label,
            string trainDataPath,
            string testDataPath,
            string evaluateFunctionName)
        {
            this.pipeline = pipeline;
            this.pipelineSweeper = pipelineSweeper;
            this.parameterSweeper = parameterSweeper;
            this.pipelineRunHistory = new List<IRunResult>();
            this.label = label;
            this.parameterSweeperIter = parameterIteration;
            this.pipelineSweeperIter = pipelineSweeperIteration;
            this.trainDataPath = trainDataPath;
            this.testDataPath = testDataPath;
            this.evaludateFunctionName = evaluateFunctionName;
            foreach (var param1 in this.pipelineSweeper.ProposeSweeps(this.pipeline, this.pipelineSweeperIter))
            {
                var singleEstiamtorPipeline = this.pipeline.BuildFromParameters(param1);
                foreach (var param2 in this.parameterSweeper.ProposeSweeps(singleEstiamtorPipeline, this.parameterSweeperIter))
                {
                    this.generateParameters.Add((singleEstiamtorPipeline.ToDataContract(), param2));
                }
            }

            this.WriteLine($"[NniTuner] Initialized");
        }

        public event EventHandler<TrialMetric> TrailMetricHandler;

        public event EventHandler<OutputEventArgs> OutputHandler;

        public string GenerateParameters(int parameterId)
        {
            try
            {
                (var contract, var param) = this.generateParameters[parameterId % this.generateParameters.Count()];
                this.WriteLine($"[NniTuner] Generated hyper Parameter #{parameterId} : {string.Join(";", param.Select(kv => $"{kv.Key}:{kv.Value}"))}");
                var trailParameter = new TrailParameter()
                {
                    Parameters = param,
                    Pipeline = contract,
                    TrainDataPath = this.trainDataPath,
                    TestDataPath = this.testDataPath,
                    Label = this.label,
                    EvaluateFunction = this.evaludateFunctionName,
                    OutputFolder = HardCode.OutputFolder
                };

                return JsonConvert.SerializeObject(trailParameter);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine(e.StackTrace);
                throw e;
            }
        }

        public void ReceiveTrialResult(TrialMetric metric)
        {
            var json = Uri.UnescapeDataString(metric.Value);
            var trailResult = JsonConvert.DeserializeObject<TrialResult>(json);
            var param = trailResult.Parameters;
            var value = trailResult.Metrics[this.evaludateFunctionName];
            var runResult = new RunResult(param, value, true);
            this.pipelineRunHistory.Add(runResult);
            this.TrailMetricHandler?.Invoke(this, metric);
            this.WriteLine($"[RandomTuner] Received Result #{metric.ParameterId} : {value}");
        }

        public void TrialEnd(int parameterId)
        {
            this.WriteLine($"[RandomTuner] Trial End #{parameterId}");
        }

        private void WriteLine(string msg)
        {
            Console.WriteLine(msg);
            this.OutputHandler?.Invoke(this, new OutputEventArgs(msg));
        }
    }

    internal struct TrailParameter
    {
        public string Label;
        public SingleEstimatorSweepablePipelineDataContract Pipeline;
        public IDictionary<string, string> Parameters;
        public string TrainDataPath;
        public string TestDataPath;
        public string EvaluateFunction;
        public string OutputFolder;
    }
}

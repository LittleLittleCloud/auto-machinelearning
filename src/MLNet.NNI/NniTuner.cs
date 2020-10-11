using MLNet.AutoPipeline;
using MLNet.Expert;
using MLNet.Expert.Contract;
using MLNet.Sweeper;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Nni {
    class NniTuner : Nni.ITuner
    {
        private ISweeper pipelineSweeper;
        private ISweeper parameterSweeper;
        private ISweepable<SingleEstimatorSweepablePipeline> pipeline;
        private List<IRunResult> pipelineRunHistory;
        private string trainDataPath = Path.GetFullPath("iris.csv");
        private string testDataPath = Path.GetFullPath("iris.csv");
        private int pipelineSweeperIter;
        private int parameterSweeperIter;
        private List<(SingleEstimatorSweepablePipelineDataContract, IDictionary<string, string>)> generateParameters = new List<(SingleEstimatorSweepablePipelineDataContract, IDictionary<string, string>)>();
        private LocalTrainingService.Option option;

        public NniTuner(ISweepable<SingleEstimatorSweepablePipeline> pipeline, ISweeper pipelineSweeper, int pipelineSweeperIteration, ISweeper parameterSweeper, LocalTrainingService.Option option)
        {
            this.pipeline = pipeline;
            this.pipelineSweeper = pipelineSweeper;
            this.parameterSweeper = parameterSweeper;
            this.pipelineRunHistory = new List<IRunResult>();
            this.option = option;
            this.parameterSweeperIter = option.ParameterSweepingIteration;
            this.pipelineSweeperIter = pipelineSweeperIteration;
            foreach (var param1 in this.pipelineSweeper.ProposeSweeps(this.pipeline, this.pipelineSweeperIter))
            {
                var singleEstiamtorPipeline = this.pipeline.BuildFromParameters(param1);
                foreach (var param2 in this.parameterSweeper.ProposeSweeps(singleEstiamtorPipeline, this.parameterSweeperIter))
                {
                    this.generateParameters.Add((singleEstiamtorPipeline.ToDataContract(), param2));
                }
            }

            Console.WriteLine($"[NniTuner] Initialized");
        }

        public string GenerateParameters(int parameterId)
        {
            try
            {
                (var contract, var param) = this.generateParameters[parameterId];
                Console.WriteLine($"[NniTuner] Generated hyper Parameter #{parameterId} : {string.Join(";", param.Select(kv => $"{kv.Key}:{kv.Value}"))}");
                var trailParameter = new TrailParameter()
                {
                    Parameters = param,
                    Pipeline = contract,
                    TrainDataPath = this.trainDataPath,
                    TestDataPath = this.testDataPath,
                    Option = this.option,
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

        public void ReceiveTrialResult(int parameterId, double metric)
        {
            var param = this.generateParameters[parameterId].Item2;
            this.pipelineRunHistory.Add(new RunResult(param, metric, true));
            Console.WriteLine($"[RandomTuner] Received Result #{parameterId} : {metric}");
        }

        public void TrialEnd(int parameterId)
        {
            Console.WriteLine($"[RandomTuner] Trial End #{parameterId}");
        }
    }

    internal struct TrailParameter
    {
        public LocalTrainingService.Option Option;
        public SingleEstimatorSweepablePipelineDataContract Pipeline;
        public IDictionary<string, string> Parameters;
        public string TrainDataPath;
        public string TestDataPath;
    }
}

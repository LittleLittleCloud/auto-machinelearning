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
        private Dictionary<int, IDictionary<string, string>> generatedParameters;
        private List<IRunResult> pipelineRunHistory;
        private string trainDataPath = Path.GetFullPath("iris.csv");
        private string testDataPath = Path.GetFullPath("iris.csv");
        private LocalTrainingService.Option option;

        public NniTuner(ISweepable<SingleEstimatorSweepablePipeline> pipeline, ISweeper pipelineSweeper, ISweeper parameterSweeper, LocalTrainingService.Option option)
        {
            this.pipeline = pipeline;
            this.pipelineSweeper = pipelineSweeper;
            this.parameterSweeper = parameterSweeper;
            this.pipelineRunHistory = new List<IRunResult>();
            this.option = option;
            this.generatedParameters = new Dictionary<int, IDictionary<string, string>>();
            Console.WriteLine($"[NniTuner] Initialized");
        }

        public string GenerateParameters(int parameterId)
        {
            try
            {
                var parameter = this.pipelineSweeper.ProposeSweeps(this.pipeline, 1).First();
                var singleEstiamtroPipeline = this.pipeline.BuildFromParameters(parameter);
                var hyperParameter = this.parameterSweeper.ProposeSweeps(singleEstiamtroPipeline, 1).First();
                this.generatedParameters.Add(parameterId, parameter);
                Console.WriteLine($"[NniTuner] Generated pipeline parameter #{parameterId} : {string.Join(";", parameter.Select(kv => $"{kv.Key}:{kv.Value}"))}");
                Console.WriteLine($"[NniTuner] Generated hyper Parameter #{parameterId} : {string.Join(";", hyperParameter.Select(kv => $"{kv.Key}:{kv.Value}"))}");
                var trailParameter = new TrailParameter()
                {
                    Parameters = hyperParameter,
                    Pipeline = singleEstiamtroPipeline.ToDataContract(),
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
            var param = this.generatedParameters[parameterId];
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

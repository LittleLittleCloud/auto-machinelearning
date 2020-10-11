using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Expert;
using MLNet.Sweeper;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Nni
{
    class Experiment
    {
        private TrainingManager.Option option;
        private SweepablePipeline pipeline;
        private MLContext context;

        public Experiment(MLContext context, TrainingManager.Option option, SweepablePipeline pipeline, string trainDataPath)
        {
            this.option = option;
            this.pipeline = pipeline;
            this.context = context;
            this.searchSpace = "-0.5,0.5";
            this.tunerName = "Random";
        }

        public async Task<(string, double)[]> Run(int trialNum, int port = 8080)
        {
            var pipelineSweeper = this.context.AutoML().Serializable().Factory.CreateSweeper(this.option.PipelineSweeper);
            var parameterSweeper = this.context.AutoML().Serializable().Factory.CreateSweeper(this.option.ParameterSweeper);
            var localOption = new LocalTrainingService.Option()
            {
                ParameterSweeper = this.option.ParameterSweeper,
                ParameterSweepingIteration = this.option.ParameterSweepingIteration,
                EvaluationMetric = this.option.EvaluationMetric,
                IsMaximizng = this.option.IsMaximizng,
                Label = this.option.Label,
                MaximumTrainingTime = this.option.MaximumTrainingTime,
                Metrics = this.option.Metrics,
            };

            this.RunTunerBackground(this.pipeline, pipelineSweeper, parameterSweeper, localOption);
            await Launch(trialNum, port);
            while (await CheckStatus() == "RUNNING")
                await Task.Delay(5000);
            var ret = await GetResults();
            Stop();
            return ret;
        }

        public void RunTunerBackground(SweepablePipeline pipeline, ISweeper pipelineSweeper, ISweeper parameterSweeper, LocalTrainingService.Option option)
        {
            var pipe = new NamedPipeServerStream(HardCode.CsPipePath, PipeDirection.InOut);
            var tuner = new NniTuner(pipeline, pipelineSweeper, parameterSweeper, option);
            dispatcher = new Nni.Dispatcher(tuner, pipe);
            dispatcher.Run();
        }

        public async Task Launch(int trialNum, int port)
        {
            Console.WriteLine("Starting NNI experiment...");
            var startInfo = new ProcessStartInfo(HardCode.NodePath)
            {
                ArgumentList = {
                    "--max-old-space-size=4096",
                    Path.Join(HardCode.NniManagerPath, "main.js"),
                    "--port", port.ToString(),
                    "--mode", "local",
                    "--start_mode", "new",
                    "--log_level", "debug",
                    "--dispatcher_pipe", HardCode.NodePipePath
                }
            };
            proc = Process.Start(startInfo);

            host = $"http://localhost:{port}/api/v1/nni";

            Console.WriteLine("Waiting REST API online...");
            while (await CheckStatus() == null)
                await Task.Delay(1000);

            Console.WriteLine("Setting up NNI manager...");
            experimentId = await SetupNniManager(trialNum);

            Console.WriteLine($"NNI experiment started (ID:{experimentId})");
        }

        public void Stop()
        {
            proc.Kill(true);
        }

        public async Task<(string, double)[]> GetResults()
        {
            var resp = await client.GetAsync(host + "/metric-data/?type=FINAL");
            string content = await resp.Content.ReadAsStringAsync();
            var metrics = JsonSerializer.Deserialize<List<MetricData>>(content, jsonOptions);

            var metricDict = new Dictionary<int, double>();
            foreach (var metric in metrics)
            {
                int paramId = Int32.Parse(metric.ParameterId);
                double result = Double.Parse(JsonSerializer.Deserialize<string>(metric.Data));
                metricDict.Add(paramId, result);
            }

            var parameters = dispatcher.GetParameters();

            var ret = new List<(string, double)>();
            for (int i = 0; i < parameters.Count; i++)
                if (metricDict.ContainsKey(i))
                    ret.Add( (parameters[i], metricDict[i]) );
            return ret.ToArray();
        }

        public async Task<string> CheckStatus()
        {
            HttpResponseMessage resp;
            try
            {
                resp = await client.GetAsync(host + "/check-status");
            }
            catch (Exception)
            {
                return null;
            }
            if (!resp.IsSuccessStatusCode)  // TODO: check if this may happen
                return null;

            string content = await resp.Content.ReadAsStringAsync();
            var status = JsonSerializer.Deserialize<NniManagerStatus>(content, jsonOptions);
            Console.WriteLine("NNI manager status: " + status.Status);
            return status.Status;
        }

        private readonly static HttpClient client = new HttpClient();
        private static JsonSerializerOptions jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        private string trialClassName;
        private string tunerName;
        private string searchSpace;
        private Dispatcher dispatcher;

        private string host; 
        private Process proc;
        private string experimentId;

        private async Task<string> SetupNniManager(int trialNum)
        {
            var trialConfig = new TrialConfig
            {
                Command = HardCode.TrialCommand + trialClassName,
                CodeDir = HardCode.TrialDir,
                GpuNum = 0
            };

            string trialConfigJson = JsonSerializer.Serialize(trialConfig, jsonOptions);
            string contentStr = "{\"trial_config\":" + trialConfigJson + "}";
            var content = new StringContent(contentStr, Encoding.UTF8, "application/json");
            var resp = await client.PutAsync(host + "/experiment/cluster-metadata", content);
            resp.EnsureSuccessStatusCode();

            var tunerConfig = new TunerConfig { BuiltinTunerName = tunerName };

            ClusterConfigKV[] clusterConfigs = {
                new ClusterConfigKV { Key = "codeDir", Value = HardCode.TrialDir },
                new ClusterConfigKV { Key = "command", Value = HardCode.TrialCommand + trialClassName }
            };

            var expConfig = new ExperimentConfig
            {
                AuthorName = "ML.NET",
                ExperimentName = "Model Builder",
                TrialConcurrency = 1,
                MaxExecDuration = 999999,
                MaxTrialNum = trialNum,
                SearchSpace = searchSpace,
                TrainingServicePlatform = "local",
                Tuner = tunerConfig,
                VersionCheck = false,
                ClusterMetaData = clusterConfigs
            };

            string expConfigJson = JsonSerializer.Serialize(expConfig, jsonOptions);
            content = new StringContent(expConfigJson, Encoding.UTF8, "application/json");
            resp = await client.PostAsync(host + "/experiment", content);
            resp.EnsureSuccessStatusCode();

            string expIdContent = await resp.Content.ReadAsStringAsync();
            var expId = JsonSerializer.Deserialize<NniExperimentId>(expIdContent);
            return expId.ExperimentId;
        }
    }

    class TrialConfig {
        public string Command { get; set; }
        public string CodeDir { get; set; }
        public int GpuNum { get; set; }
    }

    class ExperimentConfig {
        public string AuthorName { get; set; }
        public string ExperimentName { get; set; }
        public int TrialConcurrency { get; set; }
        public int MaxExecDuration { get; set; }
        public int MaxTrialNum { get; set; }
        public string SearchSpace { get; set; }
        public string TrainingServicePlatform { get; set; }
        public TunerConfig Tuner { get; set; }
        public bool VersionCheck { get; set; }
        public IList<ClusterConfigKV> ClusterMetaData { get; set; }
    }

    class TunerConfig {
        public string BuiltinTunerName { get; set; }
        public Dictionary<string, string> ClassArgs { get; set; }
    }

    class ClusterConfigKV {
        public string Key { get; set; }
        public string Value { get; set; }
    }

    class NniManagerStatus {
        public string Status { get; set; }
        public IList<string> Errors { get; set; }
    }

    class NniExperimentId {
        [JsonPropertyName("experiment_id")]
        public string ExperimentId { get; set; }
    }

    class MetricData {
        public long Timestamp { get; set; }
        public string TrialJobId { get; set; }
        public string ParameterId { get; set; }
        public string Type { get; set; }
        public int Sequence { get; set; }
        public string Data { get; set; }
    }
}

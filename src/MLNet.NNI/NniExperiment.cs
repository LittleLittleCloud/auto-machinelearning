using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Expert;
using MLNet.Sweeper;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.IO.Pipes;
using System.Net.Http;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLNet.NNI
{
    public class NniExperiment: IDisposable
    {
        private readonly HttpClient client = new HttpClient();
        private SweepablePipeline pipeline;
        private JsonSerializerSettings jsonSerializerSettings;
        private Option option;
        private string trialClassName;
        private string tunerName;
        private string searchSpace;
        private Dispatcher dispatcher;
        private Task dispatcherTask;
        private string host;
        private Process proc;
        private string experimentId;
        private bool disposedValue;

        public NniExperiment(SweepablePipeline pipeline, Option option)
        {
            this.option = option;
            this.pipeline = pipeline;
            this.searchSpace = "blablablablabla";
            this.tunerName = "GridSearch + RandomGridSearch";
            var contractResolver = new DefaultContractResolver
            {
                NamingStrategy = new CamelCaseNamingStrategy(),
            };

            this.jsonSerializerSettings = new JsonSerializerSettings
            {
                ContractResolver = contractResolver,
                StringEscapeHandling = StringEscapeHandling.EscapeHtml,
            };

            if (option.ModelOutputFolder != null)
            {
                HardCode.OutputFolder = option.ModelOutputFolder;
            }
        }

        public event EventHandler<OutputEventArgs> OutputHandler;

        public event EventHandler<TrailMetric> TrialMetricHandler;

        public async Task StartTrainingAsync(MLContext context, IDataView train, IDataView test, IProgress<TrailMetric> reporter = null, CancellationToken ct = default)
        {
            // save train and test file
            using (var stream = new FileStream(this.option.TrainPath, FileMode.Create))
            {
                context.Data.SaveAsBinary(train, stream);
            }

            using (var stream = new FileStream(this.option.TestPath, FileMode.Create))
            {
                context.Data.SaveAsBinary(test, stream);
            }

            var trailNum = this.option.PipelineSweeperIteration * this.option.ParameterSweeperIteration;
            await this.RunAsync(context, trailNum, reporter: reporter, ct: ct);
            this.WriteLine("Trial finish");
        }

        public async Task StopAsync()
        {
            this.proc.Kill();
            await this.dispatcherTask;
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            this.Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        private async Task<(string, double)[]> RunAsync(MLContext context, int trialNum, int port = 8080, IProgress<TrailMetric> reporter = null, CancellationToken ct = default)
        {
            this.dispatcherTask = this.RunTunerBackground(context, reporter);
            await this.LaunchAsync(trialNum, port);
            while (await this.CheckStatusAsync() == "RUNNING")
            {
                ct.ThrowIfCancellationRequested();
                await Task.Delay(50);
            }

            var ret = await this.GetResultsAsync();
            await this.StopAsync();
            return ret;
        }

        private Task RunTunerBackground(MLContext context, IProgress<TrailMetric> reporter = null, CancellationToken ct = default)
        {
            var pipe = new NamedPipeServerStream(HardCode.CsPipePath, PipeDirection.InOut);
            var tuner = new NniTuner(
                            this.pipeline,
                            context.AutoML().Serializable().Factory.CreateSweeper(this.option.PipelineSweeper),
                            this.option.PipelineSweeperIteration,
                            context.AutoML().Serializable().Factory.CreateSweeper(this.option.ParameterSweeper),
                            this.option.ParameterSweeperIteration,
                            this.option.Label,
                            this.option.TrainPath,
                            this.option.TestPath,
                            this.option.EvaluateFunction);
            tuner.TrailMetricHandler += this.Tuner_TrialMetricHandler;
            tuner.TrailMetricHandler += (object sender, TrailMetric e) =>
            {
                reporter?.Report(e);
            };

            tuner.OutputHandler += this.Tuner_OutputHandler;
            this.dispatcher = new Dispatcher(tuner, pipe);
            return this.dispatcher.RunAsync(ct);
        }

        private void Tuner_OutputHandler(object sender, OutputEventArgs e)
        {
            this.OutputHandler?.Invoke(sender, e);
        }

        private void Tuner_TrialMetricHandler(object sender, TrailMetric e)
        {
            this.TrialMetricHandler?.Invoke(sender, e);
        }

        private async Task LaunchAsync(int trialNum, int port)
        {
            this.WriteLine("Starting NNI experiment...");
            var startInfo = new ProcessStartInfo(HardCode.NodePath)
            {
                Arguments = string.Join(" ", new []
                        {
                            "--max-old-space-size=4096",
                            Path.Combine(HardCode.NniManagerPath, "main.js"),
                            "--port", port.ToString(),
                            "--mode", "local",
                            "--start_mode", "new",
                            "--log_level", "debug",
                            "--dispatcher_pipe", HardCode.NodePipePath,
                        }),
            };
            this.proc = Process.Start(startInfo);

            this.host = $"http://localhost:{port}/api/v1/nni";

            this.WriteLine("Waiting REST API online...");
            while (await this.CheckStatusAsync() == null)
            {
                await Task.Delay(1000);
            }

            this.WriteLine("Setting up NNI manager...");
            this.experimentId = await this.SetupNniManagerAsync(trialNum);

            this.WriteLine($"NNI experiment started (ID:{this.experimentId})");
        }

        private async Task<(string, double)[]> GetResultsAsync()
        {
            var resp = await this.client.GetAsync(this.host + "/metric-data/?type=FINAL");
            string content = await resp.Content.ReadAsStringAsync();
            var metrics = JsonConvert.DeserializeObject<List<MetricData>>(content, this.jsonSerializerSettings);

            var metricDict = new Dictionary<int, double>();
            foreach (var metric in metrics)
            {
                int paramId = int.Parse(metric.ParameterId);
                var json = Uri.UnescapeDataString(metric.Data);

                // metric.Data is wrapped between a pair of "", remove it so it can be correctly parsed.
                var result = JsonConvert.DeserializeObject<TrialResult>(json.Substring(1, json.Length - 2));
                metricDict.Add(paramId, result.Metrics["score"]);
            }

            var parameters = this.dispatcher.GetParameters();
            var ret = new List<(string, double)>();
            for (int i = 0; i < parameters.Count; i++)
            {
                if (metricDict.ContainsKey(i))
                {
                    ret.Add((parameters[i], metricDict[i]));
                }
            }

            return ret.ToArray();
        }

        private async Task<string> CheckStatusAsync()
        {
            HttpResponseMessage resp;
            try
            {
                resp = await this.client.GetAsync(this.host + "/check-status");
            }
            catch (Exception)
            {
                return null;
            }

            if (!resp.IsSuccessStatusCode) // TODO: check if this may happen
                return null;

            string content = await resp.Content.ReadAsStringAsync();
            var status = JsonConvert.DeserializeObject<NniManagerStatus>(content, this.jsonSerializerSettings);
            return status.Status;
        }

        private async Task<string> SetupNniManagerAsync(int trialNum)
        {
            var trialConfig = new TrialConfig
            {
                Command = HardCode.TrialCommand,
                CodeDir = HardCode.TrialDir,
                GpuNum = 0,
            };

            string trialConfigJson = JsonConvert.SerializeObject(trialConfig, this.jsonSerializerSettings);
            string contentStr = "{\"trial_config\":" + trialConfigJson + "}";
            var content = new StringContent(contentStr, Encoding.UTF8, "application/json");
            var resp = await this.client.PutAsync(this.host + "/experiment/cluster-metadata", content);
            resp.EnsureSuccessStatusCode();

            var tunerConfig = new TunerConfig { BuiltinTunerName = this.tunerName };

            ClusterConfigKV[] clusterConfigs =
            {
                new ClusterConfigKV { Key = "codeDir", Value = HardCode.TrialDir },
                new ClusterConfigKV { Key = "command", Value = HardCode.TrialCommand + trialClassName },
            };

            var expConfig = new ExperimentConfig
            {
                AuthorName = "ML.NET",
                ExperimentName = "Model Builder",
                TrialConcurrency = 1,
                MaxExecDuration = 999999,
                MaxTrialNum = trialNum,
                SearchSpace = this.searchSpace,
                TrainingServicePlatform = "local",
                Tuner = tunerConfig,
                VersionCheck = false,
                ClusterMetaData = clusterConfigs,
            };

            string expConfigJson = JsonConvert.SerializeObject(expConfig, this.jsonSerializerSettings);
            content = new StringContent(expConfigJson, Encoding.UTF8, "application/json");
            resp = await this.client.PostAsync(this.host + "/experiment", content);
            resp.EnsureSuccessStatusCode();

            string expIdContent = await resp.Content.ReadAsStringAsync();
            var expId = JsonConvert.DeserializeObject<NniExperimentId>(expIdContent, this.jsonSerializerSettings);
            return expId.ExperimentId;
        }

        private void WriteLine(string msg)
        {
            Console.WriteLine(msg);
            this.OutputHandler?.Invoke(this, new OutputEventArgs(msg));
        }

        public class Option
        {
            public string PipelineSweeper;

            public int PipelineSweeperIteration = 20;

            public string ParameterSweeper;

            public bool IsMaximizng;

            public string[] Metrics;

            public int ParameterSweeperIteration = 100;

            public string EvaluateFunction;

            public string Label;

            public string TrainPath { get; set; }

            public string TestPath { get; set; }

            public TaskType TaskType;

            public string ModelOutputFolder { get; set; }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!this.disposedValue)
            {
                if (disposing)
                {
                    this.StopAsync().Wait();
                }

                this.dispatcher = null;
                this.dispatcherTask = null;
                this.proc = null;
                this.disposedValue = true;
            }
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

    [DataContract]
    class NniExperimentId {
        [DataMember(Name ="experiment_id")]
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

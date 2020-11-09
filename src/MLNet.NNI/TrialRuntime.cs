using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Expert;
using MLNet.Expert.Contract;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.NNI
{
    interface ITrial
    {
        double Run(string parameter);
    }

    class TrialRuntime
    {
        public static void Run()
        {
            try
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var context = new MLContext();
                context.Log += Context_Log;
                var json = GetParameters();
                var trailParameter = JsonConvert.DeserializeObject<TrailParameter>(json);
                var outputFolder = trailParameter.OutputFolder;
                Console.WriteLine(JsonConvert.SerializeObject(trailParameter.Pipeline));
                var pipeline = trailParameter.Pipeline.ToPipeline(context);
                var parameters = trailParameter.Parameters;
                var trainDataPath = trailParameter.TrainDataPath;
                var testDataPath = trailParameter.TestDataPath;
                var train = context.Data.LoadFromBinary(trainDataPath);
                var test = context.Data.LoadFromBinary(testDataPath);
                var estiamtorChain = pipeline.BuildFromParameters(parameters);
                var model = estiamtorChain.Fit(train);
                var eval = model.Transform(test);
                Console.WriteLine(string.Join(",", eval.Preview(1).Schema.Select(column => column.Name)));
                Console.WriteLine(pipeline.ToString());
                var modelPath = Path.Combine(outputFolder, _parameterId + ".zip");
                context.Model.Save(model, train.Schema, modelPath);
                var score = context.AutoML().Serializable().Factory.CreateEvaluateFunction(trailParameter.EvaluateFunction)(context, eval, trailParameter.Label);
                stopWatch.Stop();
                var metrics = new Dictionary<string, double>();
                metrics.Add(trailParameter.EvaluateFunction, score);
                ReportResult(
                    (Dictionary<string, string>)parameters,
                    metrics,
                    stopWatch.ElapsedMilliseconds,
                    modelPath,
                    trailParameter.Pipeline);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine(e.StackTrace);
                throw e;
            }
        }

        private static void Context_Log(object sender, LoggingEventArgs e)
        {
            Console.WriteLine(e.Message);
        }

        public static string GetParameters()
        {
            Init();
            string paramStr = File.ReadAllText(Path.Combine(_sysDir, "parameter.cfg"));
            var param = JToken.Parse(paramStr).Root;
            _parameterId = (int)param.SelectToken("parameter_id");
            return (string)param.SelectToken("parameters");
        }

        public static void ReportResult(
                    Dictionary<string, string> parameters,
                    Dictionary<string, double> metrics,
                    float duration,
                    string modelPath,
                    SingleEstimatorSweepablePipelineDataContract pipelineContract)
        {
            Init();

            var trailResult = new TrialResult()
            {
                Metrics = metrics,
                Parameters = parameters,
                Duration = duration,
                ModelPath = modelPath,
                PipelineContract = pipelineContract,
            };

            var value = JsonConvert.SerializeObject(trailResult);
            var metric = new TrailMetric
            {
                ParameterId = _parameterId,
                TrialJobId = _trialJobId,
                Type = "FINAL",
                Sequence = 0,
                Value = Uri.EscapeDataString(value),
            };

            string data = JsonConvert.SerializeObject(metric) + '\n';
            string len = Encoding.ASCII.GetBytes(data).Length.ToString("D6");
            string msg = "ME" + len + data;

            string metricDir = Path.Combine(_sysDir, ".nni");
            Directory.CreateDirectory(metricDir);
            string path = Path.Combine(metricDir, "metrics");

            using (var stream = new StreamWriter(path))
            {
                stream.Write(msg);
                stream.Flush();
                _resultReported = true;
            }
        }

        private static string _sysDir = null;
        private static string _trialJobId;
        private static int _parameterId;
        private static bool _resultReported = false;

        private static void Init()
        {
            Debug.Assert(!_resultReported, "A trial can only report result once");

            if (_sysDir != null)
                return;

            _sysDir = Environment.GetEnvironmentVariable("NNI_SYS_DIR");
            Debug.Assert(_sysDir != null, "NNI trial API can only be used inside NNI trial environment");

            _trialJobId = Environment.GetEnvironmentVariable("NNI_TRIAL_JOB_ID");
        }
    }

    public class TrialResult
    {
        public Dictionary<string, double> Metrics { get; set; }

        public Dictionary<string, string> Parameters { get; set; }

        public SingleEstimatorSweepablePipelineDataContract PipelineContract { get; set; }

        public string ModelPath { get; set; }

        public float Duration { get; set; }
    }

    [DataContract]
    public class TrailMetric
    {
        [DataMember(Name = "parameter_id")]
        public int ParameterId { get; set; }

        [DataMember(Name = "trial_job_id")]
        public string TrialJobId { get; set; }

        [DataMember(Name = "type")]
        public string Type { get; set; }

        [DataMember(Name = "sequence")]
        public int Sequence { get; set; }

        [DataMember(Name = "value")]
        public string Value { get; set; }
    }
}

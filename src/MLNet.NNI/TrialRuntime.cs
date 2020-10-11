using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Expert;
using Newtonsoft.Json;
using nni_lib;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace Nni
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
                var context = new MLContext();
                context.Log += Context_Log;
                var json = GetParameters();
                var trailParameter = JsonConvert.DeserializeObject<TrailParameter>(json);
                var option = trailParameter.Option;
                Console.WriteLine(option.Label);
                Console.WriteLine(JsonConvert.SerializeObject(trailParameter.Pipeline));
                var pipeline = trailParameter.Pipeline.ToPipeline(context);
                var parameters = trailParameter.Parameters;
                var trainDataPath = trailParameter.TrainDataPath;
                var testDataPath = trailParameter.TestDataPath;
                var train = context.Data.LoadFromTextFile<Iris>(trainDataPath, hasHeader: true, separatorChar: ',');
                var test = context.Data.LoadFromTextFile<Iris>(testDataPath, hasHeader: true, separatorChar: ',');
                var estiamtorChain = pipeline.BuildFromParameters(parameters);
                var model = estiamtorChain.Fit(train);
                var eval = model.Transform(test);
                Console.WriteLine(string.Join(",", eval.Preview(1).Schema.Select(column => column.Name)));
                Console.WriteLine(pipeline.ToString());
                var score = context.AutoML().Serializable().Factory.CreateEvaluateFunction(option.EvaluationMetric)(context, eval, option.Label);
                ReportResult(score);
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
            string paramStr = File.ReadAllText(Path.Join(_sysDir, "parameter.cfg"));
            var param = JsonDocument.Parse(paramStr).RootElement;
            _parameterId = param.GetProperty("parameter_id").GetInt32();
            return param.GetProperty("parameters").GetString();
        }

        public static void ReportResult(double result)
        {
            Init();

            var metric = new TrialMetric
            {
                ParameterId = _parameterId,
                TrialJobId = _trialJobId,
                Type = "FINAL",
                Sequence = 0,
                Value = result.ToString()
            };

            string data = JsonSerializer.Serialize(metric) + '\n';
            string len = Encoding.ASCII.GetBytes(data).Length.ToString("D6");
            string msg = "ME" + len + data;

            string metricDir = Path.Join(_sysDir, ".nni");
            Directory.CreateDirectory(metricDir);
            string path = Path.Join(metricDir, "metrics");

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

    class TrialMetric
    {
        [JsonPropertyName("parameter_id")]
        public int ParameterId { get; set; }

        [JsonPropertyName("trial_job_id")]
        public string TrialJobId { get; set; }

        [JsonPropertyName("type")]
        public string Type { get; set; }

        [JsonPropertyName("sequence")]
        public int Sequence { get; set; }

        [JsonPropertyName("value")]
        public string Value { get; set; }
    }

    class Reporter : IProgress<IterationInfo>
    {
        public static Reporter Instance = new Reporter();
        public void Report(IterationInfo value)
        {
            Console.WriteLine(value.Parameters);
            Console.WriteLine($"validate score: {value.EvaluateScore}");
            Console.WriteLine($"training time: {value.TrainingTime}");
        }
    }
}

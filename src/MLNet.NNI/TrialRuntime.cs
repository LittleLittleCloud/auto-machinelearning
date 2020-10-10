using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nni
{
    interface ITrial
    {
        double Run(string parameter);
    }

    class TrialRuntime
    {
        public static void Run(string trialClassName)
        {
            var trialType = Type.GetType(trialClassName);
            Debug.Assert(typeof(ITrial).IsAssignableFrom(trialType), $"{trialClassName} is not ITrial");
            var trial = (Nni.ITrial)Activator.CreateInstance(trialType);

            ReportResult(trial.Run(GetParameters()));
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
}

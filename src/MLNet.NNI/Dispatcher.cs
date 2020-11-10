using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO.Pipes;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using MLNet.Sweeper;
using System.Runtime.Serialization;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace MLNet.NNI
{
    interface ITuner
    {
        event EventHandler<OutputEventArgs> OutputHandler;
        event EventHandler<TrailMetric> TrailMetricHandler;
        string GenerateParameters(int parameterId);
        void ReceiveTrialResult(TrailMetric metric);
        void TrialEnd(int parameterId);
    }

    class Dispatcher
    {
        private JsonSerializerSettings jsonSerializerSettings;

        public Dispatcher(ITuner tuner, NamedPipeServerStream pipe)
        {
            this.pipe = pipe;
            this.tuner = tuner;
            var contractResolver = new DefaultContractResolver
            {
                NamingStrategy = new CamelCaseNamingStrategy(),
            };

            this.jsonSerializerSettings = new JsonSerializerSettings
            {
                ContractResolver = contractResolver,
            };
        }

        public async Task RunAsync(CancellationToken ct = default)
        {
            await pipe.WaitForConnectionAsync();
            while (true) {
                if (ct.IsCancellationRequested)
                {
                    return;
                }

                (string command, string data) = await ReceiveCommand();
                if (command == null || command == "TE")  // EOF or terminate
                    break;
                HandleCommand(command, data);
            }
        }

        public List<string> GetParameters()
        {
            return parameters;
        }

        protected void UpdateSearchSpace(string data)
        {
            // do nothing
            Console.WriteLine($"Seach space: {data}");
        }

        protected void RequestTrialJobs(string data)
        {
            int num = Int32.Parse(data);
            for (int i = 0; i < num; i++) {
                string parameter = tuner.GenerateParameters(this.currentParameterId);
                var newTrialData = new NewTrialJobCommandData
                {
                    ParameterId = currentParameterId,
                    ParameterSource = "algorithm",
                    Parameters = parameter,
                    ParameterIndex = 0
                };
                string json = JsonConvert.SerializeObject(newTrialData, this.jsonSerializerSettings);
                SendCommand("TR", json);
                currentParameterId += 1;
                parameters.Add(parameter);
            }
        }

        protected void ReportMetricData(string data)
        {
            var metric = JsonConvert.DeserializeObject<TrailMetric>(data, this.jsonSerializerSettings);
            tuner.ReceiveTrialResult(metric);
        }

        protected void TrialEnd(string data)
        {
            string paramData = (string)JToken.Parse(data).SelectToken("hyper_params");
            int paramId = (int)JToken.Parse(paramData).SelectToken("parameter_id");
            tuner.TrialEnd(paramId);
        }

        private int currentParameterId = 0;
        private ITuner tuner;
        private NamedPipeServerStream pipe;
        private List<string> parameters = new List<string>();

        private void SendCommand(string command, string data)
        {
            int length = Encoding.UTF8.GetBytes(data).Length;
            string message = command + length.ToString("D14") + data;
            byte[] buffer = Encoding.UTF8.GetBytes(message);
            pipe.Write(buffer, 0, buffer.Length);
        }

        private async Task<(string, string)> ReceiveCommand()
        {
            byte[] commandBuffer = new byte[2];
            int cnt = await pipe.ReadAsync(commandBuffer, 0, 2);
            if (cnt == 0) return (null, null);  // EOF, NNI manager has been killed
            string command = Encoding.ASCII.GetString(commandBuffer);

            byte[] lengthBuffer = new byte[14];
            await pipe.ReadAsync(lengthBuffer, 0, 14);
            int length = Int32.Parse(Encoding.ASCII.GetString(lengthBuffer));

            byte[] dataBuffer = new byte[length];
            await pipe.ReadAsync(dataBuffer, 0, length);
            string data = Encoding.UTF8.GetString(dataBuffer);

            return (command, data);
        }

        private void HandleCommand(string command, string data)
        {
            if (command == "PI") {  // ping
                // do nothing

            } else if (command == "IN") {  // initialize
                UpdateSearchSpace(data);
                SendCommand("ID", "");

            } else if (command == "GE") {  // request trial jobs
                RequestTrialJobs(data);

            } else if (command == "ME") {  // report metric data
                ReportMetricData(data);

            } else if (command == "EN") {  // trial end
                TrialEnd(data);

            } else {
                // "update search space", "import data", "add customized trial job" will not happen in Model Builder
                throw new Exception($"Bad tuner command {command}");
            }
        }
    }

    [DataContract]
    class NewTrialJobCommandData
    {
        [DataMember(Name ="parameter_id")]
        public int ParameterId { get; set; }

        [DataMember(Name = "parameter_source")]
        public string ParameterSource { get; set; }

        [DataMember(Name ="parameters")]
        public string Parameters { get; set; }  // FIXME

        [DataMember(Name = "parameter_index")]
        public int ParameterIndex { get; set; }
    }

    [DataContract]
    class ReportMetricDataCommandData
    {
        [DataMember(Name = "parameter_id")]
        public int ParameterId { get; set; }

        [DataMember(Name = "trial_job_id")]
        public string TrialJobId { get; set; }
        public string Type { get; set; }
        public int Sequence { get; set; }
        public string Value { get; set; }
    }
}

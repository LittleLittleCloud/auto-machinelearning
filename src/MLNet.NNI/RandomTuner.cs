using System;

namespace Nni {
    class RandomTuner : Nni.ITuner
    {
        public void UpdateSearchSpace(string searchSpace)
        {
            Console.WriteLine($"[RandomTuner] Search Space : {searchSpace}");
            string[] split = searchSpace.Split(',');
            min = Double.Parse(split[0]);
            max = Double.Parse(split[1]);
        }

        public string GenerateParameters(int parameterId)
        {
            double ret = random.NextDouble() * (max - min) + min;
            Console.WriteLine($"[RandomTuner] Generated Parameter #{parameterId} : {ret}");
            return ret.ToString();
        }

        public void ReceiveTrialResult(int parameterId, double metric)
        {
            Console.WriteLine($"[RandomTuner] Received Result #{parameterId} : {metric}");
        }

        public void TrialEnd(int parameterId)
        {
            Console.WriteLine($"[RandomTuner] Trial End #{parameterId}");
        }

        private static Random random = new Random();

        private double min;
        private double max;
    }
}

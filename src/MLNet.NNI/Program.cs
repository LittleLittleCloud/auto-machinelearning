using System;
using System.Threading.Tasks;

using System.IO.Pipes;
using System.Text;

class Program
{
    public static async Task Main(string[] args)
    {
        if (args.Length == 0) {
            // Configure trial class, tuner, and search space to create experiment
            string trialClassName = "NaiveExample.NaiveTrial";
            string searchSpace = "-0.5,0.5";  // naive search space for POV, "min,max"
            string tuner = "Random";
            var exp = new Nni.Experiment(trialClassName, tuner, searchSpace);

            // Select number of trials to run
            int trialNum = 5;
            var result = await exp.Run(trialNum);

            // Print result
            Console.WriteLine("=== Experiment Result ===");
            foreach (var kv in result)
            {
                (string parameter, double metric) = kv;
                Console.WriteLine($"Parameter: {parameter}  Result: {metric}");
            }

        } else if (args[0] == "--trial") {
            Nni.TrialRuntime.Run(args[1]);

        } else if (args[0] == "--debug") {
            Console.WriteLine("[debug]");
        }
    }
}

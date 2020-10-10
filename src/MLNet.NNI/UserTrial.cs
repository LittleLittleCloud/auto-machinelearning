using System;
using System.Threading;

namespace NaiveExample {
    class NaiveTrial : Nni.ITrial
    {
        public double Run(string parameter)
        {
            Thread.Sleep(1000);
            double x = Double.Parse(parameter);
            return x * 2;
        }
    }
}

// <copyright file="SweepableEstimatorGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline
{
    internal class SweepableEstimatorGenerator : IDiscreteValueGenerator
    {
        public SweepableEstimatorGenerator(string name, IEnumerable<SweepableEstimatorBase> estimators)
        {
            this.Estimators = estimators.ToList();
            this.ID = Guid.NewGuid().ToString("N");
            this.Name = name;
        }

        public SweepableEstimatorGenerator(string name, SweepableEstimatorBase estimator)
        {
            this.Estimators = new List<SweepableEstimatorBase>()
            {
                estimator,
            };

            this.ID = Guid.NewGuid().ToString("N");
            this.Name = name;
        }

        public IParameterValue this[int i] => Utils.CreateObjectParameterValue(this.Name, this.Estimators[i], this.OneHotEncodeValue(this.Estimators[i]), this.ID);

        public int Count => this.Estimators.Count();

        public List<SweepableEstimatorBase> Estimators { get; private set; }

        public string Name { get; set; }

        public string ID { get; private set; }

        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            return this[(int)(this.Count * normalizedValue)];
        }

        public double[] OneHotEncodeValue(IParameterValue value)
        {
            return this.OneHotEncodeValue(value.RawValue as SweepableEstimatorBase);
        }

        private double[] OneHotEncodeValue(SweepableEstimatorBase node)
        {
            var index = this.Estimators.IndexOf(node);
            if (index < 0)
            {
                throw new Exception($"can't find value {node.EstimatorName}");
            }
            else
            {
                var onehot = Enumerable.Repeat(0.0, this.Count).ToArray();
                onehot[index] = 1;
                return onehot;
            }
        }
    }
}

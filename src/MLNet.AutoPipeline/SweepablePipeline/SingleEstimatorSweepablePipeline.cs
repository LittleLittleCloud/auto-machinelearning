// <copyright file="SingleEstimatorSweepablePipeline.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SingleEstimatorSweepablePipeline : ISweepable<EstimatorChain<ITransformer>>
    {
        internal SingleEstimatorSweepablePipeline(List<SweepableEstimatorBase> estimators)
        {
            this.Estimators = estimators;
        }

        public IEnumerable<IValueGenerator> SweepableValueGenerators
        {
            get
            {
                return this.Estimators.Select(node => node.SweepableValueGenerators)
                                      .Where(generators => generators != null)
                                      .SelectMany(x => x)
                                      .ToList();
            }
        }

        public List<SweepableEstimatorBase> Estimators { get; private set; }

        public EstimatorChain<ITransformer> BuildFromParameters(IDictionary<string, string> parameters)
        {
            var pipeline = new EstimatorChain<ITransformer>();

            for (int i = 0; i < this.Estimators.Count; i++)
            {
                if (this.Estimators[i] == SweepableEstimator<IEstimator<ITransformer>>.EmptyNode)
                {
                    continue;
                }

                pipeline = pipeline.Append(this.Estimators[i].BuildFromParameters(parameters), this.Estimators[i].Scope);
            }

            return pipeline;
        }

        public override string ToString()
        {
            return $"SweepablePipeline({string.Join("=>", this.Estimators.Select(node => node.EstimatorName))})";
        }
    }
}

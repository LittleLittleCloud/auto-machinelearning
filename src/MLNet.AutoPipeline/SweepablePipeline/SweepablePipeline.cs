// <copyright file="SweepablePipeline.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SweepablePipeline : ISweepable<SingleEstimatorSweepablePipeline>
    {
        public IEnumerable<IValueGenerator> SweepableValueGenerators { get => this.EstimatorGenerators; }

        internal IList<DiscreteValueGenerator<SweepableEstimatorBase>> EstimatorGenerators { get; private set; }

        public SweepablePipeline()
        {
            this.EstimatorGenerators = new List<DiscreteValueGenerator<SweepableEstimatorBase>>();
        }

        public SweepablePipeline Append(SweepableEstimatorBase estimator)
        {
            var i = this.EstimatorGenerators.Count();
            var option = new DiscreteValueGenerator<SweepableEstimatorBase>.Option<SweepableEstimatorBase>()
            {
                Values = new SweepableEstimatorBase[] { estimator },
                Name = $"{nameof(SweepablePipeline)}_{i}",
            };

            this.EstimatorGenerators.Add(new DiscreteValueGenerator<SweepableEstimatorBase>(option));

            return this;
        }

        public SweepablePipeline Append(params SweepableEstimatorBase[] estimators)
        {
            if (estimators.Length == 0)
            {
                return this;
            }

            var i = this.EstimatorGenerators.Count();
            var option = new DiscreteValueGenerator<SweepableEstimatorBase>.Option<SweepableEstimatorBase>()
            {
                Values = estimators,
                Name = $"{nameof(SweepablePipeline)}_{i}",
            };

            this.EstimatorGenerators.Add(new DiscreteValueGenerator<SweepableEstimatorBase>(option));

            return this;
        }

        public SweepablePipeline Append<TNewTrans>(TNewTrans estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : IEstimator<ITransformer>
        {
            var estimatorWrapper = Util.CreateSweepableEstimator(estimator, scope);
            this.Append(estimatorWrapper);

            return this;
        }

        public override string ToString()
        {
            return $"SweepablePipeline({string.Join("=>", this.EstimatorGenerators.Select(builder => $"[{string.Join("|", builder.Values.Select(node => node.EstimatorName))}]"))})";
        }

        public SingleEstimatorSweepablePipeline BuildFromParameters(IDictionary<string, string> parameters)
        {
            var estimators = new List<SweepableEstimatorBase>();

            foreach (var generator in this.EstimatorGenerators)
            {
                // TODO
                // Error Handling
                if (!parameters.ContainsKey(generator.ID))
                {
                    throw new Exception("can't build SingleSweepablePipeline from SweepablePipeline");
                }

                var valueText = parameters[generator.ID];
                var estimator = generator.CreateFromString(valueText).RawValue;

                estimators.Add(estimator as SweepableEstimatorBase);
            }

            return new SingleEstimatorSweepablePipeline(estimators);
        }
    }
}

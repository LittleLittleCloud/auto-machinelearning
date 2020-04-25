// <copyright file="SingleNodeChain.cs" company="BigMiao">
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
    public class SingleNodeChain : ISingleNodeChain
    {
        public IList<IValueGenerator> ValueGenerators { get; private set; }

        public IList<ISingleNodeBuilder> SingleNodeBuilders { get; private set; }

        public ISweeper Sweeper { get; private set; }

        public SingleNodeChain()
        {
            this.ValueGenerators = new List<IValueGenerator>();
            this.SingleNodeBuilders = new List<ISingleNodeBuilder>();
            this.Sweeper = new UniformRandomSweeper(new UniformRandomSweeper.Option());
        }

        private SingleNodeChain(IList<IValueGenerator> valueGenerators, IList<ISingleNodeBuilder> singleNodeBuilders, ISweeper sweeper)
        {
            this.ValueGenerators = valueGenerators;
            this.SingleNodeBuilders = singleNodeBuilders;
            this.Sweeper = sweeper;
        }

        public ISingleNodeChain Append(ISingleNodeBuilder builder)
        {
            this.SingleNodeBuilders.Add(builder);
            if (builder.ValueGenerators != null)
            {
                this.ValueGenerators = this.ValueGenerators.Concat(builder.ValueGenerators.ToList()).ToList();
            }

            return this;
        }

        public ISingleNodeChain Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : ITransformer
        {
            var estimatorWrapper = new EstimatorWrapper<TNewTrans>(estimator, scope);
            this.Append(estimatorWrapper);

            return this;
        }

        public string Summary()
        {
            return $"SingleNodeChain({string.Join("=>", this.SingleNodeBuilders.Select(builder => builder.EstimatorName))})";
        }

        public override string ToString()
        {
            return this.Summary();
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this.Sweeper = sweeper;
            this.Sweeper.SweepableParamaters = this.ValueGenerators;
        }

        public IEnumerable<EstimatorChain<ITransformer>> Sweeping(int maximum)
        {
            // index of autoEstimator
            foreach (var parameters in this.Sweeper.ProposeSweeps(maximum))
            {

                var pipeline = new EstimatorChain<ITransformer>();
                for (int i = 0; i < this.SingleNodeBuilders.Count; i++)
                {
                    pipeline = pipeline.Append(this.SingleNodeBuilders[i].BuildEstimator(parameters), this.SingleNodeBuilders[i].Scope);
                }

                yield return pipeline;
            }
        }

        public ISingleNodeChain Concat(ISingleNodeChain chain)
        {
            return new SingleNodeChain(this.ValueGenerators.Concat(chain.ValueGenerators).ToList(), this.SingleNodeBuilders.Concat(chain.SingleNodeBuilders).ToList(), this.Sweeper);
        }
    }
}

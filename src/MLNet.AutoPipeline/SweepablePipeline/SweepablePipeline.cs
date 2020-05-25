// <copyright file="SweepablePipeline.cs" company="BigMiao">
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
    public class SweepablePipeline : ISweepablePipeline
    {
        public IList<IValueGenerator> ValueGenerators { get; private set; }

        public IList<ISweepablePipelineNode> SingleNodeBuilders { get; private set; }

        public ISweeper Sweeper { get; private set; }

        public SweepablePipeline()
        {
            this.ValueGenerators = new List<IValueGenerator>();
            this.SingleNodeBuilders = new List<ISweepablePipelineNode>();
            this.Sweeper = new UniformRandomSweeper(new UniformRandomSweeper.Option());
        }

        private SweepablePipeline(IList<IValueGenerator> valueGenerators, IList<ISweepablePipelineNode> singleNodeBuilders, ISweeper sweeper)
        {
            this.ValueGenerators = valueGenerators;
            this.SingleNodeBuilders = singleNodeBuilders;
            this.Sweeper = sweeper;
        }

        public ISweepablePipeline Append(ISweepablePipelineNode builder)
        {
            this.SingleNodeBuilders.Add(builder);
            if (builder.ValueGenerators != null)
            {
                this.ValueGenerators = this.ValueGenerators.Concat(builder.ValueGenerators.ToList()).ToList();
            }

            return this;
        }

        public ISweepablePipeline Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : ITransformer
        {
            var estimatorWrapper = new UnsweepableNode<TNewTrans>(estimator, scope);
            this.Append(estimatorWrapper);

            return this;
        }

        public string Summary()
        {
            return $"SweepablePipeline({string.Join("=>", this.SingleNodeBuilders.Select(builder => builder.EstimatorName))})";
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
            if (this.ValueGenerators.Count == 0)
            {
                var pipeline = new EstimatorChain<ITransformer>();
                for (int i = 0; i < this.SingleNodeBuilders.Count; i++)
                {
                    if (this.SingleNodeBuilders[i] == UnsweepableNode<ITransformer>.EmptyNode)
                    {
                        continue;
                    }

                    pipeline = pipeline.Append(this.SingleNodeBuilders[i].BuildEstimator(), this.SingleNodeBuilders[i].Scope);
                }

                yield return pipeline;
            }
            else
            {
                foreach (var parameters in this.Sweeper.ProposeSweeps(maximum))
                {
                    yield return this.BuildFromParameterSet(parameters);
                }
            }
        }

        public ISweepablePipeline Concat(ISweepablePipeline chain)
        {
            return new SweepablePipeline(this.ValueGenerators.Concat(chain.ValueGenerators).ToList(), this.SingleNodeBuilders.Concat(chain.SingleNodeBuilders).ToList(), this.Sweeper);
        }

        public ISweepablePipeline Append<TNewTrains, TOption>(Func<TOption, IEstimator<TNewTrains>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains : ITransformer
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTrains, TOption>(estimatorBuilder, optionBuilder, scope);
            this.Append(autoEstimator);
            return this;
        }

        internal EstimatorChain<ITransformer> BuildFromParameterSet(ParameterSet parameters)
        {
            var pipeline = new EstimatorChain<ITransformer>();
            for (int i = 0; i < this.SingleNodeBuilders.Count; i++)
            {
                if (this.SingleNodeBuilders[i] == UnsweepableNode<ITransformer>.EmptyNode)
                {
                    continue;
                }

                pipeline = pipeline.Append(this.SingleNodeBuilders[i].BuildEstimator(parameters), this.SingleNodeBuilders[i].Scope);
            }

            return pipeline;
        }
    }
}

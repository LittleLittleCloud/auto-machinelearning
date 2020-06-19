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

        public IList<ISweepablePipelineNode> SweepablePipelineNodes { get; private set; }

        public ISweeper Sweeper { get; private set; }

        public SweepablePipeline()
        {
            this.ValueGenerators = new List<IValueGenerator>();
            this.SweepablePipelineNodes = new List<ISweepablePipelineNode>();
            this.Sweeper = new UniformRandomSweeper(new UniformRandomSweeper.Option());
        }

        private SweepablePipeline(IList<IValueGenerator> valueGenerators, IList<ISweepablePipelineNode> singleNodeBuilders, ISweeper sweeper)
        {
            this.ValueGenerators = valueGenerators;
            this.SweepablePipelineNodes = singleNodeBuilders;
            this.Sweeper = sweeper;
        }

        public ISweepablePipeline Append(ISweepablePipelineNode builder)
        {
            this.SweepablePipelineNodes.Add(builder);
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
            return $"SweepablePipeline({string.Join("=>", this.SweepablePipelineNodes.Select(builder => builder.EstimatorName))})";
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

        public IEnumerable<SweepingInfo> Sweeping(int maximum)
        {
            if (this.ValueGenerators.Count == 0)
            {
                var pipeline = new EstimatorChain<ITransformer>();
                for (int i = 0; i < this.SweepablePipelineNodes.Count; i++)
                {
                    if (this.SweepablePipelineNodes[i] == UnsweepableNode<ITransformer>.EmptyNode)
                    {
                        continue;
                    }

                    pipeline = pipeline.Append(this.SweepablePipelineNodes[i].BuildEstimator(), this.SweepablePipelineNodes[i].Scope);
                }

                yield return new SweepingInfo(pipeline, null);
            }
            else
            {
                foreach (var parameters in this.Sweeper.ProposeSweeps(maximum))
                {
                    yield return new SweepingInfo(this.BuildFromParameterSet(parameters), parameters);
                }
            }
        }

        public ISweepablePipeline Concat(ISweepablePipeline chain)
        {
            return new SweepablePipeline(this.ValueGenerators.Concat(chain.ValueGenerators).ToList(), this.SweepablePipelineNodes.Concat(chain.SweepablePipelineNodes).ToList(), this.Sweeper);
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
            for (int i = 0; i < this.SweepablePipelineNodes.Count; i++)
            {
                if (this.SweepablePipelineNodes[i] == UnsweepableNode<ITransformer>.EmptyNode)
                {
                    continue;
                }

                pipeline = pipeline.Append(this.SweepablePipelineNodes[i].BuildEstimator(parameters), this.SweepablePipelineNodes[i].Scope);
            }

            return pipeline;
        }
    }
}

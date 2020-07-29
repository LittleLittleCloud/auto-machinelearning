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
    public class SweepablePipeline
    {
        public IList<IValueGenerator> ValueGenerators { get; private set; }

        public IList<INode> Nodes { get; private set; }

        public ISweeper Sweeper { get; private set; }

        public SweepablePipeline()
        {
            this.ValueGenerators = new List<IValueGenerator>();
            this.Nodes = new List<INode>();
            this.Sweeper = new UniformRandomSweeper(new UniformRandomSweeper.Option());
        }

        private SweepablePipeline(IList<IValueGenerator> valueGenerators, IList<INode> singleNodeBuilders, ISweeper sweeper)
        {
            this.ValueGenerators = valueGenerators;
            this.Nodes = singleNodeBuilders;
            this.Sweeper = sweeper;
        }

        public SweepablePipeline Append<TTrain>(INode<TTrain> builder)
            where TTrain: IEstimator<ITransformer>
        {
            return this.Append((INode)builder);
        }

        public SweepablePipeline Append(INode builder)
        {
            this.Nodes.Add(builder);
            if (builder.ValueGenerators != null)
            {
                this.ValueGenerators = this.ValueGenerators.Concat(builder.ValueGenerators.ToList()).ToList();
            }

            return this;
        }

        public SweepablePipeline Append<TNewTrans>(TNewTrans estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : IEstimator<ITransformer>
        {
            var estimatorWrapper = new UnsweepableNode<TNewTrans>(estimator, scope);
            this.Append(estimatorWrapper);

            return this;
        }

        public string Summary()
        {
            return $"SweepablePipeline({string.Join("=>", this.Nodes.Select(builder => builder.EstimatorName))})";
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
                for (int i = 0; i < this.Nodes.Count; i++)
                {
                    if (this.Nodes[i] == UnsweepableNode<IEstimator<ITransformer>>.EmptyNode)
                    {
                        continue;
                    }

                    pipeline = pipeline.Append(this.Nodes[i].BuildEstimator(), this.Nodes[i].Scope);
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

        public SweepablePipeline Concat(SweepablePipeline chain)
        {
            return new SweepablePipeline(this.ValueGenerators.Concat(chain.ValueGenerators).ToList(), this.Nodes.Concat(chain.Nodes).ToList(), this.Sweeper);
        }

        public SweepablePipeline Append<TNewTrains, TOption>(Func<TOption, TNewTrains> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains : IEstimator<ITransformer>
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTrains, TOption>(estimatorBuilder, optionBuilder, scope);
            this.Append(autoEstimator);
            return this;
        }

        internal EstimatorChain<ITransformer> BuildFromParameterSet(ParameterSet parameters)
        {
            var pipeline = new EstimatorChain<ITransformer>();
            for (int i = 0; i < this.Nodes.Count; i++)
            {
                if (this.Nodes[i] == UnsweepableNode<IEstimator<ITransformer>>.EmptyNode)
                {
                    continue;
                }

                pipeline = pipeline.Append(this.Nodes[i].BuildEstimator(parameters), this.Nodes[i].Scope);
            }

            return pipeline;
        }
    }
}

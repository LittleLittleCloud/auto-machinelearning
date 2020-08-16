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
    public class SweepablePipeline : ISweepable<SingleSweepablePipeline>
    {
        public IEnumerable<IValueGenerator> SweepableValueGenerators { get; private set; }

        internal IList<SweepableNodeGenerator> NodeGenerators { get; private set; }

        public SweepablePipeline()
        {
            this.SweepableValueGenerators = new List<IValueGenerator>();
            this.NodeGenerators = new List<SweepableNodeGenerator>();
        }

        SingleSweepablePipeline ISweepable<SingleSweepablePipeline>.BuildFromParameterSet(ParameterSet parameters)
        {
            var nodes = new List<INode>();

            foreach (var generator in this.NodeGenerators)
            {
                var param = parameters.Where(param => generator.ID == param.ID).FirstOrDefault();

                // TODO
                // Error Handling
                if (param == null)
                {
                    throw new Exception("can't build SingleSweepablePipeline from SweepablePipeline");
                }

                nodes.Add(param.RawValue as INode);
            }

            return new SingleSweepablePipeline(nodes);
        }

        public SweepablePipeline Append<TTrain>(INode<TTrain> builder)
            where TTrain: IEstimator<ITransformer>
        {
            return this.Append((INode)builder);
        }

        public SweepablePipeline Append(INode node)
        {
            var i = this.NodeGenerators.Count();
            this.NodeGenerators.Add(new SweepableNodeGenerator($"{nameof(SweepableNodeGenerator)}_{i}", node));
            if (node.ValueGenerators != null)
            {
                this.SweepableValueGenerators = this.SweepableValueGenerators.Concat(node.ValueGenerators.ToList()).ToList();
            }

            return this;
        }

        public SweepablePipeline Append(params INode[] nodes)
        {
            var i = this.NodeGenerators.Count();
            this.NodeGenerators.Add(new SweepableNodeGenerator($"{nameof(SweepableNodeGenerator)}_{i}", nodes));

            return this;
        }

        public SweepablePipeline Append<TNewTrans>(TNewTrans estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : IEstimator<ITransformer>
        {
            var estimatorWrapper = Util.CreateUnSweepableNode(estimator, scope);
            this.Append(estimatorWrapper);

            return this;
        }

        public string Summary()
        {
            return $"SweepablePipeline({string.Join("=>", this.NodeGenerators.Select(builder => builder.Name))})";
        }

        public override string ToString()
        {
            return this.Summary();
        }

        internal SweepablePipeline Append(SweepableNodeGenerator nodeGenerator)
        {
            this.NodeGenerators.Add(nodeGenerator);

            return this;
        }

        internal SingleSweepablePipeline BuildFromParameterSet(ParameterSet parameters)
        {
            return (this as ISweepable<SingleSweepablePipeline>).BuildFromParameterSet(parameters);
        }
    }

    internal class SingleSweepablePipeline : ISweepable<EstimatorChain<ITransformer>>
    {
        private List<INode> nodes;

        public SingleSweepablePipeline(List<INode> nodes)
        {
            this.nodes = nodes;
        }

        public IEnumerable<IValueGenerator> SweepableValueGenerators
        {
            get
            {
                return this.nodes.Select(node => node.ValueGenerators).SelectMany(x => x).ToList();
            }
        }

        public EstimatorChain<ITransformer> BuildFromParameterSet(ParameterSet parameters)
        {
            var pipeline = new EstimatorChain<ITransformer>();
            for (int i = 0; i < this.nodes.Count; i++)
            {
                if (this.nodes[i] == UnsweepableNode<IEstimator<ITransformer>>.EmptyNode)
                {
                    continue;
                }

                pipeline = pipeline.Append(this.nodes[i].BuildFromParameterSet(parameters), this.nodes[i].Scope);
            }

            return pipeline;
        }
    }
}

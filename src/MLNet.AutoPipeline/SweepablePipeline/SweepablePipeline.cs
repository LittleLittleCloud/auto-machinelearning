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
    public class SweepablePipeline : ISweepable<SingleEstimatorSweepablePipeline>
    {
        public IEnumerable<IValueGenerator> SweepableValueGenerators { get => this.NodeGenerators; }

        internal IList<SweepableNodeGenerator> NodeGenerators { get; private set; }

        public SweepablePipeline()
        {
            this.NodeGenerators = new List<SweepableNodeGenerator>();
        }

        SingleEstimatorSweepablePipeline ISweepable<SingleEstimatorSweepablePipeline>.BuildFromParameterSet(ParameterSet parameters)
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

            return new SingleEstimatorSweepablePipeline(nodes);
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
            return $"SweepablePipeline({string.Join("=>", this.NodeGenerators.Select(builder => $"[{string.Join("|", builder.Nodes.Select(node => node.EstimatorName))}]"))})";
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

        internal SingleEstimatorSweepablePipeline BuildFromParameterSet(ParameterSet parameters)
        {
            return (this as ISweepable<SingleEstimatorSweepablePipeline>).BuildFromParameterSet(parameters);
        }
    }
}

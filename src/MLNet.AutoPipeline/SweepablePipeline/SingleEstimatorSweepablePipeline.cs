// <copyright file="SingleSweepablePipeline.cs" company="BigMiao">
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

    internal class SingleEstimatorSweepablePipeline : ISweepable<EstimatorChain<ITransformer>>
    {
        private List<INode> nodes;

        public SingleEstimatorSweepablePipeline(List<INode> nodes)
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

        public List<INode> Nodes { get => this.nodes; }

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

        public override string ToString()
        {
            return $"SweepablePipeline({string.Join("=>", this.nodes.Select(node => node.EstimatorName))})";
        }
    }
}

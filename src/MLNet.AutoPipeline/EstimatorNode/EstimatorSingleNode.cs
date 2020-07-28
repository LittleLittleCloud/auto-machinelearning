// <copyright file="EstimatorSingleNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class EstimatorSingleNode : IEstimatorNode
    {
        private INode estimatorBuilder;
        private static EstimatorSingleNode no_op = new EstimatorSingleNode();

        internal static EstimatorSingleNode EmptyNode
        {
            get
            {
                return no_op;
            }
        }

        public EstimatorSingleNode(INode estimatorBuilder)
        {
            this.estimatorBuilder = estimatorBuilder;
        }

        public EstimatorSingleNode(INode<IEstimator<ITransformer>> estimatorBuilder)
        {
            this.estimatorBuilder = (INode)estimatorBuilder;
        }

        private EstimatorSingleNode()
        {
            this.estimatorBuilder = (INode)UnsweepableNode<IEstimator<ITransformer>>.EmptyNode;
        }

        public EstimatorNodeType NodeType => EstimatorNodeType.Node;

        public IEnumerable<ISweepablePipeline> BuildSweepablePipelines()
        {
            yield return new SweepablePipeline().Append(this.estimatorBuilder);
        }

        public string Summary()
        {
            return $"SingleNode({this.estimatorBuilder.EstimatorName})";
        }

        public override string ToString()
        {
            return this.Summary();
        }
    }
}

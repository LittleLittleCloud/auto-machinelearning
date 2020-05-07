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
        private ISweepablePipelineNode estimatorBuilder;
        private static EstimatorSingleNode no_op = new EstimatorSingleNode();

        internal static EstimatorSingleNode EmptyNode
        {
            get
            {
                return no_op;
            }
        }

        public EstimatorSingleNode(ISweepablePipelineNode estimatorBuilder)
        {
            this.estimatorBuilder = estimatorBuilder;
        }

        private EstimatorSingleNode()
        {
            this.estimatorBuilder = UnsweepableNode<ITransformer>.EmptyNode;
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

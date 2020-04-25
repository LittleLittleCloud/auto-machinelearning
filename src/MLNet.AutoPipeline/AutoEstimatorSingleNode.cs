// <copyright file="AutoEstimatorSingleNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class AutoEstimatorSingleNode : IAutoEstimatorNode
    {
        private ISingleNodeBuilder estimatorBuilder;

        public AutoEstimatorSingleNode(ISingleNodeBuilder estimatorBuilder)
        {
            this.estimatorBuilder = estimatorBuilder;
        }

        public AutoEstimatorNodeType NodeType => AutoEstimatorNodeType.Node;

        public IEnumerable<ISingleNodeChain> BuildEstimatorChains()
        {
            yield return new SingleNodeChain().Append(this.estimatorBuilder);
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

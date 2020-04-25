// <copyright file="EstimatorSingleNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class EstimatorSingleNode : IEstimatorNode
    {
        private ISweepablePipelineNode estimatorBuilder;

        public EstimatorSingleNode(ISweepablePipelineNode estimatorBuilder)
        {
            this.estimatorBuilder = estimatorBuilder;
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

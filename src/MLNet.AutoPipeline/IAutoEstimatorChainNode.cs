// <copyright file="IEstimatorChainBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MLNet.AutoPipeline
{
    public enum AutoEstimatorChainNodeType
    {
        /// <summary>
        /// Single Node. ie: IEstimator.
        /// </summary>
        SingleNode = 0,

        /// <summary>
        /// AutoEstimatorChain type.
        /// </summary>
        AutoEstimatorChain = 1,

        /// <summary>
        /// array of SingleNode|AutoEstimatorChain.
        /// </summary>
        MixedNode = 2,
    };

    public interface IAutoEstimatorChainNode
    {
        IEnumerable<IEnumerable<IEstimatorBuilder>> BuildEstimatorChains();

        AutoEstimatorChainNodeType NodeType { get; }

        string Summary();
    }

    public class AutoEstimatorSingleNode : IAutoEstimatorChainNode
    {
        private IEstimatorBuilder estimatorBuilder;

        public AutoEstimatorSingleNode(IEstimatorBuilder estimatorBuilder)
        {
            this.estimatorBuilder = estimatorBuilder;
        }

        public AutoEstimatorChainNodeType NodeType => AutoEstimatorChainNodeType.SingleNode;

        public IEnumerable<IEnumerable<IEstimatorBuilder>> BuildEstimatorChains()
        {
            yield return new List<IEstimatorBuilder>() { this.estimatorBuilder };
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

    public class AutoEstimatorMixedNode : IAutoEstimatorChainNode
    {
        private IEnumerable<IAutoEstimatorChainNode> _nodes;

        public AutoEstimatorMixedNode(IEnumerable<IAutoEstimatorChainNode> nodes)
        {
            this._nodes = nodes;
        }

        public AutoEstimatorChainNodeType NodeType => AutoEstimatorChainNodeType.MixedNode;

        public IEnumerable<IEnumerable<IEstimatorBuilder>> BuildEstimatorChains()
        {
            foreach (var node in this._nodes)
            {
                foreach (var chain in node.BuildEstimatorChains())
                {
                    yield return chain;
                }
            }
        }

        public string Summary()
        {
            return $"MixedNode({string.Join(", ", this._nodes.Select(node => node.Summary()))})";
        }

        public override string ToString()
        {
            return this.Summary();
        }
    }
}

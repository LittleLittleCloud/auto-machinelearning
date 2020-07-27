// <copyright file="EstimatorNodeGroup.cs" company="BigMiao">
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
    public class EstimatorNodeGroup : IEstimatorNode
    {
        private IList<IEstimatorNode> _nodes;

        public EstimatorNodeGroup(IEnumerable<IEstimatorNode> nodes)
        {
            foreach (var node in nodes)
            {
                if (node.NodeType == EstimatorNodeType.NodeGroup)
                {
                    throw new Exception("NodeGroup can only contain Node and NodeChain type");
                }
            }

            this._nodes = nodes.ToList();
        }

        public EstimatorNodeGroup()
        {
            this._nodes = new List<IEstimatorNode>();
        }

        public EstimatorNodeType NodeType => EstimatorNodeType.NodeGroup;

        public IEnumerable<ISweepablePipeline> BuildSweepablePipelines()
        {
            foreach (var node in this._nodes)
            {
                foreach (var chain in node.BuildSweepablePipelines())
                {
                    yield return chain;
                }
            }
        }

        public EstimatorNodeGroup Append<TTrans>(TTrans estimator, TransformerScope scope = TransformerScope.Everything)
            where TTrans : IEstimator<ITransformer>
        {
            var node = new UnsweepableNode<TTrans>(estimator, scope);

            return this.Append(new EstimatorSingleNode(node));
        }

        public EstimatorNodeGroup Append(EstimatorSingleNode node)
        {
            this._nodes.Add(node);
            return this;
        }

        public EstimatorNodeGroup Append(EstimatorNodeChain node)
        {
            this._nodes.Add(node);
            return this;
        }

        public EstimatorNodeGroup Append<TNewTran, TOption>(Func<TOption, TNewTran> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTran : IEstimator<ITransformer>
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTran, TOption>(estimatorBuilder, optionBuilder, scope);
            this.Append(new EstimatorSingleNode(autoEstimator));
            return this;
        }

        /// <summary>
        /// Append an empty <see cref="EstimatorSingleNode"/> to <see cref="EstimatorNodeGroup"/>.
        /// </summary>
        public EstimatorNodeGroup OrNone()
        {
            this.Append(EstimatorSingleNode.EmptyNode);
            return this;
        }

        public string Summary()
        {
            return $"NodeGroup({string.Join(", ", this._nodes.Select(node => node.Summary()))})";
        }

        public override string ToString()
        {
            return this.Summary();
        }
    }
}

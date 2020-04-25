// <copyright file="AutoEstimatorNodeGroup.cs" company="BigMiao">
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

    public class AutoEstimatorNodeGroup : IAutoEstimatorNode
    {
        private IList<IAutoEstimatorNode> _nodes;

        public AutoEstimatorNodeGroup(IEnumerable<IAutoEstimatorNode> nodes)
        {
            foreach (var node in nodes)
            {
                if (node.NodeType == AutoEstimatorNodeType.NodeGroup)
                {
                    throw new Exception("NodeGroup can only contain Node and NodeChain type");
                }
            }

            this._nodes = nodes.ToList();
        }

        public AutoEstimatorNodeGroup()
        {
            this._nodes = new List<IAutoEstimatorNode>();
        }

        public AutoEstimatorNodeType NodeType => AutoEstimatorNodeType.NodeGroup;

        public IEnumerable<ISingleNodeChain> BuildEstimatorChains()
        {
            foreach (var node in this._nodes)
            {
                foreach (var chain in node.BuildEstimatorChains())
                {
                    yield return chain;
                }
            }
        }

        public AutoEstimatorNodeGroup Append(AutoEstimatorSingleNode node)
        {
            this._nodes.Add(node);
            return this;
        }

        public AutoEstimatorNodeGroup Append(AutoEstimatorNodeChain node)
        {
            this._nodes.Add(node);
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

// <copyright file="EstimatorNodeChain.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class EstimatorNodeChain : IEstimatorNode
    {
        private readonly IList<IEstimatorNode> _nodes;

        private ISweeper _sweeper;

        public EstimatorNodeType NodeType => EstimatorNodeType.NodeChain;

        public EstimatorNodeChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
            : this()
        {
            for (int i = 0; i != estimators.Length; ++i)
            {
                var estimatorWrapper = new UnsweepableNode<ITransformer>(estimators[i], scopes[i]);
                this.Append(estimatorWrapper);
            }
        }

        public EstimatorNodeChain(IList<IEstimatorNode> nodes)
        {
            this._nodes = nodes;
        }

        public EstimatorNodeChain()
        {
            this._nodes = new List<IEstimatorNode>();
        }

        public EstimatorNodeChain Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : ITransformer
        {
            var estimatorWrapper = new UnsweepableNode<TNewTrans>(estimator, scope);
            return this.Append(estimatorWrapper);
        }

        public EstimatorNodeChain Append<TNewTrains, TOption>(Func<TOption, IEstimator<TNewTrains>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains : ITransformer
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTrains, TOption>(estimatorBuilder, optionBuilder, scope);

            return this.Append(autoEstimator);
        }

        public EstimatorNodeChain Append<TNewTrans, TOption>(SweepableNode<TNewTrans, TOption> estimatorBuilder)
            where TNewTrans : ITransformer
            where TOption : class
        {
            return this.Append(new EstimatorSingleNode(estimatorBuilder));
        }

        public EstimatorNodeChain Append<TNewTrans>(UnsweepableNode<TNewTrans> estimatorWrapper)
            where TNewTrans : ITransformer
        {
            return this.Append(new EstimatorSingleNode(estimatorWrapper));
        }

        public EstimatorNodeChain Append(IEstimatorNode node)
        {
            this._nodes.Add(node);

            return this;
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this._sweeper = sweeper;
        }

        public IEnumerable<ISweepablePipeline> BuildSweepablePipelines()
        {
            if (this._nodes.Count == 0)
            {
                return new List<ISweepablePipeline>();
            }

            // TODO: use stack and yield to save memory.
            var paths = new List<ISweepablePipeline>();
            foreach (var node in this._nodes)
            {
                var newPath = new List<ISweepablePipeline>();

                if (paths.Count == 0)
                {
                    foreach (var _path in node.BuildSweepablePipelines())
                    {
                        paths.Add(_path);
                    }
                }
                else
                {
                    foreach (var _path in node.BuildSweepablePipelines())
                    {
                        foreach (var _existPath in paths)
                        {
                            newPath.Add(_existPath.Concat(_path));
                        }
                    }

                    paths = newPath;
                }
            }

            return paths;
        }

        public string Summary()
        {
            return $"NodeChain({string.Join("=>", this._nodes.Select(node => node.Summary())) })";
        }
    }
}

// <copyright file="AutoEstimatorNodeChain.cs" company="BigMiao">
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
    public class AutoEstimatorNodeChain : IAutoEstimatorNode
    {
        private readonly IList<TransformerScope> _scopes;
        private readonly IList<ISingleNodeBuilder> _estimatorBuilders;
        private readonly IList<IAutoEstimatorNode> _nodes;

        private ISweeper _sweeper;

        public AutoEstimatorNodeType NodeType => AutoEstimatorNodeType.NodeChain;

        public AutoEstimatorNodeChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            this._estimatorBuilders = estimators is null ? new List<ISingleNodeBuilder>() : estimators.Select(x => new EstimatorWrapper<ITransformer>(x) as ISingleNodeBuilder).ToList();
            this._scopes = scopes is null ? new List<TransformerScope>() : scopes.ToList();
        }

        public AutoEstimatorNodeChain(IList<IAutoEstimatorNode> nodes)
        {
            this._nodes = nodes;
        }

        public AutoEstimatorNodeChain()
        {
            this._estimatorBuilders = new List<ISingleNodeBuilder>();
            this._scopes = new List<TransformerScope>();
            this._nodes = new List<IAutoEstimatorNode>();
        }

        public IEnumerable<EstimatorChain<ITransformer>> ProposePipelines(int batch)
        {
            // index of autoEstimator
            foreach (var parameters in this._sweeper.ProposeSweeps(batch))
            {

                var pipeline = new EstimatorChain<ITransformer>();
                var xfs = new ITransformer[this._estimatorBuilders.Count];
                for (int i = 0; i < this._estimatorBuilders.Count; i++)
                {
                    pipeline = pipeline.Append(this._estimatorBuilders[i].BuildEstimator(parameters), this._scopes[i]);
                }

                yield return pipeline;
            }
        }

        public AutoEstimatorNodeChain Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : ITransformer
        {
            var estimatorWrapper = new EstimatorWrapper<TNewTrans>(estimator, scope);
            this._estimatorBuilders.Add(estimatorWrapper);
            this._scopes.Add(scope);
            this._nodes.Add(new AutoEstimatorSingleNode(estimatorWrapper));

            return this;
        }

        public AutoEstimatorNodeChain Append<TNewTrains, TOption>(Func<TOption, IEstimator<TNewTrains>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains : ITransformer
            where TOption : class
        {
            var autoEstimator = new EstimatorBuilder<TNewTrains, TOption>(estimatorBuilder, optionBuilder, scope);

            this._estimatorBuilders.Add(autoEstimator);
            this._scopes.Add(scope);
            this._nodes.Add(new AutoEstimatorSingleNode(autoEstimator));
            return this;
        }

        public AutoEstimatorNodeChain Append<TNewTrans, TOption>(EstimatorBuilder<TNewTrans, TOption> estimatorBuilder)
            where TNewTrans : ITransformer
            where TOption : class
        {
            this._estimatorBuilders.Add(estimatorBuilder);
            this._scopes.Add(estimatorBuilder.Scope);
            this._nodes.Add(new AutoEstimatorSingleNode(estimatorBuilder));

            return this;
        }

        public AutoEstimatorNodeChain Append(IAutoEstimatorNode node)
        {
            this._nodes.Add(node);

            return this;
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this._sweeper = sweeper;
        }

        public IEnumerable<ISingleNodeChain> BuildEstimatorChains()
        {
            if (this._nodes.Count == 0)
            {
                return new List<ISingleNodeChain>();
            }

            // TODO: use stack and yield to save memory.
            var paths = new List<ISingleNodeChain>();
            foreach (var node in this._nodes)
            {
                var newPath = new List<ISingleNodeChain>();

                if (paths.Count == 0)
                {
                    foreach (var _path in node.BuildEstimatorChains())
                    {
                        paths.Add(_path);
                    }
                }
                else
                {
                    foreach (var _path in node.BuildEstimatorChains())
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
            return $"AutoEstimatorChain({string.Join("->", this._nodes.Select(node => node.Summary())) })";
        }
    }
}

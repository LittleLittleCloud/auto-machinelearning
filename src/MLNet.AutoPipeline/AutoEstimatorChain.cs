// <copyright file="AutoEstimatorChain.cs" company="BigMiao">
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
    public class AutoEstimatorChain : IAutoEstimatorChainNode
    {
        private readonly IList<TransformerScope> _scopes;
        private readonly IList<IEstimatorBuilder> _estimatorBuilders;
        private readonly IList<IAutoEstimatorChainNode> _nodes;

        private ISweeper _sweeper;

        public AutoEstimatorChainNodeType NodeType => AutoEstimatorChainNodeType.AutoEstimatorChain;

        public AutoEstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            this._estimatorBuilders = estimators is null ? new List<IEstimatorBuilder>() : estimators.Select(x => new EstimatorWrapper<ITransformer>(x) as IEstimatorBuilder).ToList();
            this._scopes = scopes is null ? new List<TransformerScope>() : scopes.ToList();
        }

        public AutoEstimatorChain(IList<IAutoEstimatorChainNode> nodes)
        {
            this._nodes = nodes;
        }

        public AutoEstimatorChain()
        {
            this._estimatorBuilders = new List<IEstimatorBuilder>();
            this._scopes = new List<TransformerScope>();
            this._nodes = new List<IAutoEstimatorChainNode>();
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

        public AutoEstimatorChain Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : ITransformer
        {
            var estimatorWrapper = new EstimatorWrapper<TNewTrans>(estimator, scope);
            this._estimatorBuilders.Add(estimatorWrapper);
            this._scopes.Add(scope);
            this._nodes.Add(new AutoEstimatorSingleNode(estimatorWrapper));

            return this;
        }

        public AutoEstimatorChain Append<TNewTrains, TOption>(Func<TOption, IEstimator<TNewTrains>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains: ITransformer
            where TOption: class
        {
            var autoEstimator = new EstimatorBuilder<TNewTrains, TOption>(estimatorBuilder, optionBuilder, scope);

            this._estimatorBuilders.Add(autoEstimator);
            this._scopes.Add(scope);
            this._nodes.Add(new AutoEstimatorSingleNode(autoEstimator));
            return this;
        }

        public AutoEstimatorChain Append<TNewTrans, TOption>(EstimatorBuilder<TNewTrans, TOption> estimatorBuilder)
            where TNewTrans : ITransformer
            where TOption : class
        {
            this._estimatorBuilders.Add(estimatorBuilder);
            this._scopes.Add(estimatorBuilder.Scope);
            this._nodes.Add(new AutoEstimatorSingleNode(estimatorBuilder));

            return this;
        }

        public AutoEstimatorChain Append(IAutoEstimatorChainNode node)
        {
            this._nodes.Add(node);

            return this;
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this._sweeper = sweeper;
        }

        public IEnumerable<IEnumerable<IEstimatorBuilder>> BuildEstimatorChains()
        {
            if (this._nodes.Count == 0)
            {
                return new List<List<IEstimatorBuilder>>();
            }

            // TODO: use stack and yield to save memory.
            IEnumerable<IEnumerable<IEstimatorBuilder>> paths = new List<List<IEstimatorBuilder>>();
            foreach (var node in this._nodes)
            {
                var newPath = new List<List<IEstimatorBuilder>>();
                foreach (var _path in node.BuildEstimatorChains())
                {
                    foreach (var _existPath in paths)
                    {
                        newPath.Add(_existPath.Concat(_path).ToList());
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

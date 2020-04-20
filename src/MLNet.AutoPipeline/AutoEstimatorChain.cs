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
    public class AutoEstimatorChain
    {
        private readonly IList<TransformerScope> _scopes;
        private readonly IList<IEstimatorBuilder> _estimatorBuilders;

        private ISweeper _sweeper;

        public AutoEstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            this._estimatorBuilders = estimators is null ? new List<IEstimatorBuilder>() : estimators.Select(x => new EstimatorWrapper(x) as IEstimatorBuilder).ToList();
            this._scopes = scopes is null ? new List<TransformerScope>() : scopes.ToList();
        }

        public AutoEstimatorChain()
        {
            this._estimatorBuilders = new List<IEstimatorBuilder>();
            this._scopes = new List<TransformerScope>();
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
            where TNewTrans : class, ITransformer
        {
            this._estimatorBuilders.Add(new EstimatorWrapper(estimator));
            this._scopes.Add(scope);

            return this;
        }

        public AutoEstimatorChain Append<TNewTrains, TOption>(Func<TOption, IEstimator<TNewTrains>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTrains: ITransformer
            where TOption: class
        {
            var autoEstimator = new EstimatorBuilder<TNewTrains, TOption>(estimatorBuilder, optionBuilder);

            this._estimatorBuilders.Add(autoEstimator);
            this._scopes.Add(scope);

            return this;
        }

        public AutoEstimatorChain Append<TNewTrans, TOption>(EstimatorBuilder<TNewTrans, TOption> estimatorBuilder,  TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : class, ITransformer
            where TOption : class
        {
            this._estimatorBuilders.Add(estimatorBuilder);
            this._scopes.Add(scope);

            return this;
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this._sweeper = sweeper;
        }
    }
}

// <copyright file="AutoEstimatorChain.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal class AutoEstimatorChain<TLastTransformer>
        where TLastTransformer : class, ITransformer
    {
        public readonly IEstimator<TLastTransformer> LastEstimator;
        private readonly IList<TransformerScope> _scopes;
        private readonly IList<IEstimator<ITransformer>> _estimators;

        private ISweeper _sweeper;

        public AutoEstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            this._estimators = estimators is null ? new List<IEstimator<ITransformer>>() : estimators.ToList();
            this._scopes = scopes is null ? new List<TransformerScope>() : scopes.ToList();
            this.LastEstimator = estimators.LastOrDefault() as IEstimator<TLastTransformer>;
        }

        public AutoEstimatorChain()
        {
            this._estimators = new IEstimator<ITransformer>[0];
            this._scopes = new TransformerScope[0];
            this.LastEstimator = null;
        }

        public IEnumerable<EstimatorChain<ITransformer>> ProposePipelines(int batch)
        {
            // index of autoEstimator
            var autoEstimator = this._estimators.Where(_est => _est is IAutoEstimator).FirstOrDefault() as IAutoEstimator;

            foreach (var parameters in this._sweeper.ProposeSweeps(batch))
            {
                autoEstimator.Current = parameters;

                var pipeline = new EstimatorChain<ITransformer>();
                var xfs = new ITransformer[this._estimators.Count];
                for (int i = 0; i < this._estimators.Count; i++)
                {
                    pipeline = pipeline.Append(this._estimators[i], this._scopes[i]);

                    //var est = this._estimators[i];
                    //xfs[i] = est.Fit(current);
                    //current = xfs[i].Transform(current);
                }

                yield return pipeline;
            }
        }

        public AutoEstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : class, ITransformer
        {
            this._estimators.Add(estimator);
            this._scopes.Add(scope);
            return new AutoEstimatorChain<TNewTrans>(this._estimators.ToArray(), this._scopes.ToArray());
        }

        public void UseSweeper(ISweeper sweeper)
        {
            this._sweeper = sweeper;
        }
    }
}

// <copyright file="AutoEstimatorChain.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.AutoPipeline
{
    internal class AutoEstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly TransformerScope[] _scopes;
        private readonly IEstimator<ITransformer>[] _estimators;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        public AutoEstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            this._estimators = estimators ?? new IEstimator<ITransformer>[0];
            this._scopes = scopes ?? new TransformerScope[0];
            this.LastEstimator = estimators.LastOrDefault() as IEstimator<TLastTransformer>;
        }

        public AutoEstimatorChain()
        {
            this._estimators = new IEstimator<ITransformer>[0];
            this._scopes = new TransformerScope[0];
            this.LastEstimator = null;
        }

        public TransformerChain<TLastTransformer> Fit(IDataView input)
        {
            GetOutputSchema(SchemaShape.Create(input.Schema));

            while (true)
            {

            }
        }

        public IEnumerable<(TransformerChain<TLastTransformer>, ISweeper)> Fits(IDataView input)
        {
            GetOutputSchema(SchemaShape.Create(input.Schema));

            // index of autoEstimator
            var autoEstimator = this._estimators.Where(_est => _est is IAutoEstimator).FirstOrDefault() as IAutoEstimator;
            while (true)
            {
                // check sweeper
                if (autoEstimator.Sweeper.MoveNext() == false)
                {
                    yield break;
                }

                IDataView current = input;
                var xfs = new ITransformer[this._estimators.Length];
                for (int i = 0; i < this._estimators.Length; i++)
                {
                    var est = this._estimators[i];
                    xfs[i] = est.Fit(current);
                    current = xfs[i].Transform(current);
                }

                yield return (new TransformerChain<TLastTransformer>(xfs, this._scopes), autoEstimator.Sweeper);
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var s = inputSchema;
            foreach (var est in this._estimators)
            {
                s = est.GetOutputSchema(s);
            }

            return s;
        }

        public AutoEstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : class, ITransformer
        {
            return new AutoEstimatorChain<TNewTrans>(this._estimators.AppendElement(estimator), this._scopes.AppendElement(scope));
        }
    }
}

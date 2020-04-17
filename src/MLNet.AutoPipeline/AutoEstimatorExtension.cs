// <copyright file="AutoEstimatorExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace MLNet.AutoPipeline
{
    internal static class AutoEstimatorExtension
    {
        public static AutoEstimatorChain<TNewTrain>
            Append<TLastTrain, TNewTrain, TOption>(this EstimatorChain<TLastTrain> estimatorChain,
                                                   Func<TOption, IEstimator<TNewTrain>> estimatorBuilder,
                                                   OptionBuilder<TOption> parameters,
                                                   ISweeper sweeper,
                                                   TransformerScope scope = TransformerScope.Everything)
            where TLastTrain : class, ITransformer
            where TNewTrain : class, ITransformer
            where TOption : class
        {
            var autoEstimator = new AutoEstimator<TNewTrain, TOption>(estimatorBuilder, parameters, sweeper);

            return new AutoEstimatorChain<TLastTrain>(estimatorChain.GetEstimators, estimatorChain.GetScopes)
                       .Append(autoEstimator, scope);
        }
    }
}

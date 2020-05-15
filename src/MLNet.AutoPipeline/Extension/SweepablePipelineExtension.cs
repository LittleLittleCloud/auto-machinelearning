// <copyright file="SweepablePipelineExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline.Extension
{
    public static class SweepablePipelineExtension
    {
        public static ISweepablePipeline
            Append<TLastTrain, TNewTrain, TOption>(
                this EstimatorChain<TLastTrain> estimatorChain,
                Func<TOption, IEstimator<TNewTrain>> estimatorBuilder,
                OptionBuilder<TOption> parameters,
                TransformerScope scope = TransformerScope.Everything)
            where TLastTrain : class, ITransformer
            where TNewTrain : class, ITransformer
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTrain, TOption>(estimatorBuilder, parameters, scope);

            var estimators = estimatorChain.GetType().GetField("_estimators", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(estimatorChain) as IEstimator<ITransformer>[];
            var scopes = estimatorChain.GetType().GetField("_scopes", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(estimatorChain) as TransformerScope[];

            var singleNodeChain = new SweepablePipeline();
            for (int i = 0; i != estimators.Length - 1; ++i)
            {
                var estimator = new UnsweepableNode<ITransformer>(estimators[i], scopes[i]);
                singleNodeChain.Append(estimator);
            }

            singleNodeChain.Append(new UnsweepableNode<TLastTrain>(estimators.Last() as IEstimator<TLastTrain>, scopes.Last()));

            return singleNodeChain.Append(autoEstimator);
        }

        public static ISweepablePipeline
            Append<TLastTran, TNewTran, TOption>(
                this IEstimator<TLastTran> estimator,
                Func<TOption, IEstimator<TNewTran>> estimatorBuilder,
                OptionBuilder<TOption> parameters,
                TransformerScope scope = TransformerScope.Everything)
            where TNewTran : ITransformer
            where TLastTran : ITransformer
            where TOption : class
        {
            var autoEstimator = new SweepableNode<TNewTran, TOption>(estimatorBuilder, parameters, scope);

            return new SweepablePipeline()
                        .Append(new UnsweepableNode<TLastTran>(estimator as IEstimator<TLastTran>))
                        .Append(autoEstimator);
        }

    }
}

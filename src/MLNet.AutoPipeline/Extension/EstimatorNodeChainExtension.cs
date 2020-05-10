// <copyright file="EstimatorNodeChainExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;

namespace MLNet.AutoPipeline.Extension
{
    public static class EstimatorNodeChainExtension
    {
        public static EstimatorNodeChain
            Append<TLastTrain>(
                this EstimatorChain<TLastTrain> estimatorChain,
                IEstimatorNode node)
            where TLastTrain : class, ITransformer
        {
            var estimators = estimatorChain.GetType().GetField("_estimators", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(estimatorChain) as IEstimator<ITransformer>[];
            var scopes = estimatorChain.GetType().GetField("_scopes", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).GetValue(estimatorChain) as TransformerScope[];

            return new EstimatorNodeChain(estimators, scopes)
                       .Append(node);
        }
    }
}

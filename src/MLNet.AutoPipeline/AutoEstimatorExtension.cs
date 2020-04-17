﻿// <copyright file="AutoEstimatorExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoPipeline
{
    internal static class AutoEstimatorExtension
    {
        public static AutoEstimatorChain<TNewTrain>
            Append<TLastTrain, TNewTrain, TOption>(this EstimatorChain<TLastTrain> estimatorChain,
                                                   Func<TOption, IEstimator<TNewTrain>> estimatorBuilder,
                                                   OptionBuilder<TOption> parameters,
                                                   ISweeper sweeper,
                                                   TransformerScope scope = TransformerScope.Everything)
            where TLastTrain: class, ITransformer
            where TNewTrain: class, ITransformer
            where TOption: class
        {
            var autoEstimator = new AutoEstimator<TNewTrain, TOption>(estimatorBuilder, parameters, sweeper);

            return new AutoEstimatorChain<TLastTrain>(estimatorChain.GetEstimators, estimatorChain.GetScopes)
                       .Append(autoEstimator, scope);
        }
    }
}

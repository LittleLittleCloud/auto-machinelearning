// <copyright file="Util.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal static class Util
    {
        public static SweepableNode<TTransformer, TOption> CreateSweepableNode<TTransformer, TOption>(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
            where TTransformer : ITransformer
            where TOption : class
        {
            return new SweepableNode<TTransformer, TOption>(estimatorFactory, optionBuilder, scope, estimatorName);
        }
    }
}

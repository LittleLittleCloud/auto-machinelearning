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
        private static Random rng = new Random();  
        
        public static SweepableNode<TTransformer, TOption> CreateSweepableNode<TTransformer, TOption>(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
            where TTransformer : ITransformer
            where TOption : class
        {
            return new SweepableNode<TTransformer, TOption>(estimatorFactory, optionBuilder, scope, estimatorName);
        }

        public static UnsweepableNode<TInstance> BuildConcatFeaturesTransformer<TInstance>(IEstimator<TInstance> instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
            where TInstance : ITransformer
        {
            return new UnsweepableNode<TInstance>(instance, scope, estimatorName);
        }

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}

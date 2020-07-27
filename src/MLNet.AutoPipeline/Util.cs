// <copyright file="Util.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline
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

        public static UnsweepableNode<TInstance> CreateUnSweepableNode<TInstance>(TInstance instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
            where TInstance : IEstimator<ITransformer>
        {
            return new UnsweepableNode<TInstance>(instance, scope, estimatorName, inputs, outputs);
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

        public static IEnumerable<T> PickN<T>(this IEnumerable<T> list, int n)
        {
            Contract.Requires(n >= 0 && n <= list.Count());
            var pickIndex = Enumerable.Range(0, list.Count()).ToList();
            pickIndex.Shuffle();
            return pickIndex.GetRange(0, n).Select(i => list.ToArray()[i]);
        }
    }
}

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

        public static SweepableEstimator<TNewTrain, TOption> CreateSweepableEstimator<TNewTrain, TOption>(Func<TOption, TNewTrain> estimatorFactory, SweepableOption<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
            where TNewTrain : IEstimator<ITransformer>
            where TOption : class
        {
            return new SweepableEstimator<TNewTrain, TOption>(estimatorFactory, optionBuilder, scope, estimatorName, inputs, outputs);
        }

        public static SweepableEstimator<TInstance> CreateSweepableEstimator<TInstance>(TInstance instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
            where TInstance : IEstimator<ITransformer>
        {
            return new SweepableEstimator<TInstance>(instance, scope, estimatorName, inputs, outputs);
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

        public static void CopyFieldsTo<T, TU>(this T source, TU dest)
        {
            var sourceField = typeof(T).GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance).ToList();
            var destField = typeof(TU).GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance).ToList();

            foreach (var field in sourceField)
            {
                if (destField.Any(x => x.Name == field.Name && x.FieldType == field.FieldType))
                {
                    var p = destField.First(x => x.Name == field.Name && x.FieldType == field.FieldType);
                    p.SetValue(dest, field.GetValue(source));
                }
            }
        }
    }
}

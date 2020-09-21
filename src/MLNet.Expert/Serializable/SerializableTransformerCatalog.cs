// <copyright file="SerializableTransformerCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class SerializableTransformerCatalog
    {
        public SerializableTransformerCatalog(MLContext context)
        {
            this.Context = context;
            this.Categorical = new SerializableCategoricalCatalog(context);
            this.Conversion = new SerializableConversionCatalog(context);
            this.Text = new SerializableTextCatalog(context);
        }

        public SerializableCategoricalCatalog Categorical { get; private set; }

        public SerializableConversionCatalog Conversion { get; private set; }

        public SerializableTextCatalog Text { get; }

        public MLContext Context { get; private set; }

        public SweepableEstimatorBase ReplaceMissingValues(string inputColumnName, string outputColumnName)
        {
            var instance = this.Context.Transforms.ReplaceMissingValues(outputColumnName, inputColumnName);
            return this.Context.AutoML().CreateUnsweepableEstimator(instance, new string[] { inputColumnName }, new string[] { outputColumnName }, nameof(MissingValueReplacingEstimator));
        }

        public SweepableEstimatorBase Concatnate(string[] inputColumnNames, string outputColumnName)
        {
            var instance = this.Context.Transforms.Concatenate(outputColumnName, inputColumnNames);
            return this.Context.AutoML().CreateUnsweepableEstimator(instance, inputColumnNames, new string[] { outputColumnName }, nameof(ColumnConcatenatingEstimator));
        }
    }
}

// <copyright file="SerializableCategoricalCatalog.cs" company="BigMiao">
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
    internal class SerializableCategoricalCatalog
    {
        public SerializableCategoricalCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; private set; }

        public SweepableEstimatorBase OneHotEncoding(string inputColumnName, string outputColumnName)
        {
            var onehotEncode = this.Context.Transforms.Categorical.OneHotEncoding(outputColumnName, inputColumnName);
            return this.Context.AutoML().CreateUnsweepableEstimator(onehotEncode, new string[] { inputColumnName }, new string[] { outputColumnName }, nameof(OneHotEncodingEstimator));
        }
    }
}

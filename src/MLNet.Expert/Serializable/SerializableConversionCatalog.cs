// <copyright file="SerializableConversionCatalog.cs" company="BigMiao">
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
    internal class SerializableConversionCatalog
    {
        public SerializableConversionCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; private set; }

        public SweepableEstimatorBase MapValueToKey(string inputColumnName, string outputColumnName)
        {
            var instance = this.Context.Transforms.Conversion.MapValueToKey(outputColumnName, inputColumnName);

            return this.Context.AutoML().CreateUnsweepableEstimator(instance, new string[] { inputColumnName }, new string[] { outputColumnName }, nameof(ValueToKeyMappingEstimator));
        }
    }
}

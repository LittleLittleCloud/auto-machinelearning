// <copyright file="SerializableRegressionTrainerCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class SerializableRegressionTrainerCatalog
    {
        public SerializableRegressionTrainerCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; private set; }

        public SweepableEstimatorBase LightGbm(LightGbmRegressionTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().Regression.LightGbm(label, feature, option);
        }
    }
}

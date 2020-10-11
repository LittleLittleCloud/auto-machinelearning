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

        public SweepableEstimatorBase LightGbm(string label, string feature)
        {
            var option = LightGbmRegressionTrainerSweepableOptions.Default;
            option.FeatureColumnName = feature;
            option.LabelColumnName = label;
            return this.Context.AutoML().Regression.LightGbm(label, feature, option);
        }

        public SweepableEstimatorBase Sdca(string label, string feature)
        {
            var option = SdcaRegressionTrainerSweepableOptions.Default;
            option.FeatureColumnName = feature;
            option.LabelColumnName = label;
            return this.Context.AutoML().Regression.Sdca(label, feature, option);
        }
    }
}

// <copyright file="SerializableBinaryClassificationTrainerCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class SerializableBinaryClassificationTrainerCatalog
    {
        public SerializableBinaryClassificationTrainerCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; private set; }

        public SweepableEstimatorBase LinearSvm(LinearSvmBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.LinearSvm(label, feature, option);
        }

        public SweepableEstimatorBase LdSvm(LdSvmBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.LdSvm(label, feature, option);
        }

        public SweepableEstimatorBase FastForest(FastForestBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.FastForest(label, feature, option);
        }

        public SweepableEstimatorBase FastTree(FastTreeBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.FastTree(label, feature, option);
        }

        public SweepableEstimatorBase LightGbm(LightGbmBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.LightGbm(label, feature, option);
        }

        public SweepableEstimatorBase Gam(GamBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.Gam(label, feature, option);
        }

        public SweepableEstimatorBase SgdNonCalibrated(SgdNonCalibratedBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.SgdNonCalibrated(label, feature, option);
        }

        public SweepableEstimatorBase SgdCalibrated(SgdCalibratedBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.SgdCalibrated(label, feature, option);
        }

        public SweepableEstimatorBase AveragedPerceptron(AveragedPerceptronBinaryTrainerSweepableOptions option)
        {
            var label = option.LabelColumnName.ValueGenerator[0].ValueText;
            var feature = option.FeatureColumnName.ValueGenerator[0].ValueText;
            return this.Context.AutoML().BinaryClassification.AveragedPerceptron(label, feature, option);
        }
    }
}

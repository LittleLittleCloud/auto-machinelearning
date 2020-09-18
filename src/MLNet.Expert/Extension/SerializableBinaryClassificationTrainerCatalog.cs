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

        public SweepableEstimatorBase LinearSvm(string label, string feature)
        {
            var option = LinearSvmBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.LinearSvm(label, feature, option);
        }

        public SweepableEstimatorBase LdSvm(string label, string feature)
        {
            var option = LdSvmBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.LdSvm(label, feature, option);
        }

        public SweepableEstimatorBase FastForest(string label, string feature)
        {
            var option = FastForestBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.FastForest(label, feature, option);
        }

        public SweepableEstimatorBase FastTree(string label, string feature)
        {
            var option = FastTreeBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.FastTree(label, feature, option);
        }

        public SweepableEstimatorBase LightGbm(string label, string feature)
        {
            var option = LightGbmBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.LightGbm(label, feature, option);
        }

        public SweepableEstimatorBase Gam(string label, string feature)
        {
            var option = GamBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.Gam(label, feature, option);
        }

        public SweepableEstimatorBase SgdNonCalibrated(string label, string feature)
        {
            var option = SgdNonCalibratedBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.SgdNonCalibrated(label, feature, option);
        }

        public SweepableEstimatorBase SgdCalibrated(string label, string feature)
        {
            var option = SgdCalibratedBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.SgdCalibrated(label, feature, option);
        }

        public SweepableEstimatorBase AveragedPerceptron(string label, string feature)
        {
            var option = AveragedPerceptronBinaryTrainerSweepableOptions.Default;
            option.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(feature);
            option.LabelColumnName = ParameterFactory.CreateDiscreteParameter(label);
            return this.Context.AutoML().BinaryClassification.AveragedPerceptron(label, feature, option);
        }
    }
}

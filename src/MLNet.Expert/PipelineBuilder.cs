// <copyright file="PipelineBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using MLNet.Expert.Extension;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.Expert
{
    public enum TaskType
    {
        BinaryClassification = 0,

        MultiClassification = 1,

        Regression = 2,
    }

    public class PipelineBuilder
    {
        public PipelineBuilder(TaskType taskType, bool isAzureAttach = false, bool isUsingSingleFeatureTrainer = false)
        {
            this.PipelineBuilderOption = new Option()
            {
                TaskType = taskType,
                IsAzureAttach = isAzureAttach,
                IsUsingSingleFeatureTrainer = isUsingSingleFeatureTrainer,
            };
        }

        public Option PipelineBuilderOption { get; private set; }

        public SweepablePipeline BuildPipeline(MLContext context, IEnumerable<Column> columns)
        {
            var sweepablePipeline = new SweepablePipeline();

            foreach (var column in columns)
            {
                switch (column.ColumnPurpose)
                {
                    case ColumnPurpose.NumericFeature:
                        sweepablePipeline.Append(this.GetSuggestedNumericColumnTransformers(context, column).ToArray());
                        break;
                    case ColumnPurpose.CategoricalFeature:
                        sweepablePipeline.Append(this.GetSuggestedCatagoricalColumnTransformers(context, column).ToArray());
                        break;
                    case ColumnPurpose.Label:
                        sweepablePipeline.Append(this.GetSuggestedLabelColumnTransformers(context, column).ToArray());
                        break;
                    default:
                        break;
                }
            }

            var featureColumns = columns.Where(c => c.ColumnPurpose == ColumnPurpose.CategoricalFeature
                                                 || c.ColumnPurpose == ColumnPurpose.NumericFeature
                                                 || c.ColumnPurpose == ColumnPurpose.TextFeature)
                                        .Select(c => c.Name)
                                        .ToArray();

            if (this.PipelineBuilderOption.IsUsingSingleFeatureTrainer)
            {
                sweepablePipeline.Append(context.AutoML().Serializable().Transformer.Concatnate(featureColumns, "_FEATURE"));
                var labelColumn = columns.Where(c => c.ColumnPurpose == ColumnPurpose.Label).First();
                sweepablePipeline.Append(this.GetSuggestedSingleFeatureTrainers(context, labelColumn, "_FEATURE").ToArray());
            }

            return sweepablePipeline;
        }

        public IEnumerable<SweepableEstimatorBase> GetSuggestedNumericColumnTransformers(MLContext context, Column column)
        {
            return new SweepableEstimatorBase[]
                    {
                        context.AutoML().Serializable().Transformer.ReplaceMissingValues(column.Name, column.Name),
                    };
        }

        public IEnumerable<SweepableEstimatorBase> GetSuggestedLabelColumnTransformers(MLContext context, Column column)
        {
            if (this.PipelineBuilderOption.TaskType == TaskType.MultiClassification)
            {
                return new SweepableEstimatorBase[]
                {
                    context.AutoML().Serializable().Transformer.Conversion.MapValueToKey(column.Name, column.Name),
                };
            }

            return new SweepableEstimatorBase[0];
        }

        public IEnumerable<SweepableEstimatorBase> GetSuggestedCatagoricalColumnTransformers(MLContext context, Column column)
        {
            return new SweepableEstimatorBase[]
                    {
                        context.AutoML().Serializable().Transformer.Categorical.OneHotEncoding(column.Name, column.Name),
                    };
        }

        public IEnumerable<SweepableEstimatorBase> GetSuggestedSingleFeatureTrainers(MLContext context, Column column, string featureColumnName)
        {
            switch (this.PipelineBuilderOption.TaskType)
            {
                case TaskType.BinaryClassification:
                    var res = new List<SweepableEstimatorBase>();

                    var linearSvmOption = LinearSvmBinaryTrainerSweepableOptions.Default;
                    linearSvmOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    linearSvmOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.LinearSvm(linearSvmOption));

                    var ldSvmOption = LdSvmBinaryTrainerSweepableOptions.Default;
                    ldSvmOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    ldSvmOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.LdSvm(ldSvmOption));

                    var ffOption = FastForestBinaryTrainerSweepableOptions.Default;
                    ffOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    ffOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.FastForest(ffOption));

                    var ftOption = FastTreeBinaryTrainerSweepableOptions.Default;
                    ftOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    ftOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.FastTree(ftOption));

                    var lightGbmOption = LightGbmBinaryTrainerSweepableOptions.Default;
                    lightGbmOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    lightGbmOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.LightGbm(lightGbmOption));

                    var gamOption = GamBinaryTrainerSweepableOptions.Default;
                    gamOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    gamOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.Gam(gamOption));

                    var sgdNonCalibratedOption = SgdNonCalibratedBinaryTrainerSweepableOptions.Default;
                    sgdNonCalibratedOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    sgdNonCalibratedOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.SgdNonCalibrated(sgdNonCalibratedOption));

                    var averagedPerceptronOption = AveragedPerceptronBinaryTrainerSweepableOptions.Default;
                    averagedPerceptronOption.FeatureColumnName = ParameterFactory.CreateDiscreteParameter(featureColumnName);
                    averagedPerceptronOption.LabelColumnName = ParameterFactory.CreateDiscreteParameter(column.Name);
                    res.Add(context.AutoML().Serializable().BinaryClassification.AveragedPerceptron(averagedPerceptronOption));

                    return res;
                default:
                    throw new NotImplementedException();
            }
        }

        public class Option
        {
            public TaskType TaskType { get; set; }

            public bool IsAzureAttach { get; set; }

            public bool IsUsingSingleFeatureTrainer { get; set; }
        }
    }
}

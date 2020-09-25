// <copyright file="PipelineBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
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
                    case ColumnPurpose.TextFeature:
                        sweepablePipeline.Append(this.GetSuggestedTextColumnTransformers(context, column).ToArray());
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

        public IEnumerable<SweepableEstimatorBase> GetSuggestedTextColumnTransformers(MLContext context, Column column)
        {
            return new SweepableEstimatorBase[]
            {
                context.AutoML().Serializable().Transformer.Text.FeaturizeText(column.Name, column.Name),
                context.AutoML().Serializable().Transformer.Text.FeaturizeTextWithWordEmbedding(column.Name, column.Name),
            };
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
                    res.Add(context.AutoML().Serializable().BinaryClassification.LinearSvm(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.LdSvm(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.FastForest(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.FastTree(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.LightGbm(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.Gam(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.SgdNonCalibrated(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.SgdCalibrated(column.Name, featureColumnName));
                    res.Add(context.AutoML().Serializable().BinaryClassification.AveragedPerceptron(column.Name, featureColumnName));

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

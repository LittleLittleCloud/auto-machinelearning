// <copyright file="TextCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class SerializableTextCatalog
    {
        public SerializableTextCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; }

        public SweepableEstimatorBase FeaturizeText(string inputColumn, string outputColumn)
        {
            var instance = this.Context.Transforms.Text.FeaturizeText(outputColumn, inputColumn);
            return this.Context.AutoML().CreateUnsweepableEstimator(instance, new string[] { inputColumn }, new string[] { outputColumn }, nameof(TextFeaturizingEstimator));
        }

        public SweepableEstimatorBase FeaturizeTextWithWordEmbedding(string inputColumn, string outputColumn)
        {
            var option = TextFeaturizerWithWordEmbeddingSweepableOption.Default;
            option.InputColumnName = inputColumn;
            option.OutputColumnName = outputColumn;
            return this.Context.AutoML().CreateSweepableEstimator(
                                    (context, option) =>
                                    {
                                        return context.Transforms.Text.NormalizeText(
                                                option.OutputColumnName,
                                                option.InputColumnName,
                                                option.CaseMode,
                                                option.KeepDiacritics,
                                                option.KeepPunctuations,
                                                option.KeepNumbers)
                                            .Append(context.Transforms.Text.TokenizeIntoWords(outputColumn, outputColumn))
                                            .Append(context.Transforms.Text.RemoveDefaultStopWords(outputColumn, outputColumn))
                                            .Append(context.Transforms.Text.ApplyWordEmbedding(
                                                option.OutputColumnName,
                                                option.OutputColumnName,
                                                option.ModelKind));
                                    },
                                    option,
                                    new string[] { inputColumn },
                                    new string[] { outputColumn },
                                    nameof(this.FeaturizeTextWithWordEmbedding));
        }
    }
}

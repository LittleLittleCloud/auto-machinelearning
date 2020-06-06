// <copyright file="Program.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Extension;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;

namespace MLNet.Examples.SentimentAnalysis
{
    public class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            context.Log += Context_Log;
            // Load Data
            var trainDataset = context.Data.LoadFromTextFile<ModelInput>(@".\datasets\wikipedia-detox-250-line-data-train.tsv", hasHeader: true);
            var testDataset = context.Data.LoadFromTextFile<ModelInput>(@".\datasets\wikipedia-detox-250-line-test.tsv", hasHeader: true);

            var normalizeTextOption = new NormalizeTextOption();
            var applyWordEmbeddingOption = new ApplyWordEmbeddingOption();
            var sdcaOption = new SdcaLogisticRegressionOption();

            // Create pipeline
            var pipelines = context.Transforms.Conversion.MapValueToKey("Sentiment-key", "Sentiment")
                           .Append( // Create NormalizeText transformer and sweep over it.
                               (NormalizeTextOption option) =>
                               {
                                   return context.Transforms.Text.NormalizeText(
                                       option.outputColumnName,
                                       option.inputColumnName,
                                       option.caseMode,
                                       option.keepDiacritics,
                                       option.keepPunctuations,
                                       option.keepNumbers);
                               },
                               normalizeTextOption)
                           .Append(context.Transforms.Text.TokenizeIntoWords("txt", "txt"))
                           .Append(context.Transforms.Text.RemoveDefaultStopWords("txt", "txt"))
                           .Append( // Create ApplyWordEmbedding transformer and sweep over it
                               (ApplyWordEmbeddingOption option) =>
                               {
                                   return context.Transforms.Text.ApplyWordEmbedding(
                                       option.outputColumnName,
                                       option.inputColumnName,
                                       option.modelKind);
                               },
                               applyWordEmbeddingOption)
                           .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression, sdcaOption);

            // Set up Sweeper Option
            var valueGeneratorsList = new List<IValueGenerator>();
            valueGeneratorsList.AddRange(normalizeTextOption.ValueGenerators);
            valueGeneratorsList.AddRange(applyWordEmbeddingOption.ValueGenerators);
            valueGeneratorsList.AddRange(sdcaOption.ValueGenerators);

            var sweeperOption = new UniformRandomSweeper.Option();

            var sweeper = new UniformRandomSweeper(sweeperOption);

            pipelines.UseSweeper(sweeper);

            foreach (var sweepingInfo in pipelines.Sweeping(100))
            {
                var pipeline = sweepingInfo.Pipeline;
                Console.WriteLine(sweeper.Current.ToString());

                var eval = pipeline.Fit(trainDataset).Transform(testDataset);
                var metrics = context.BinaryClassification.Evaluate(eval, "Sentiment");

                Console.WriteLine($"Accuracy: {metrics.Accuracy}");
                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
                Console.WriteLine(string.Empty);
            }

        }

        private static void Context_Log(object sender, LoggingEventArgs e)
        {
            Console.WriteLine(e.Message);
        }

        public class NormalizeTextOption : OptionBuilder<NormalizeTextOption>
        {
            public string outputColumnName = "txt";

            public string inputColumnName = "SentimentText";

            [Parameter(new object[] { TextNormalizingEstimator.CaseMode.Lower, TextNormalizingEstimator.CaseMode.None, TextNormalizingEstimator.CaseMode.Upper})]
            public TextNormalizingEstimator.CaseMode caseMode = TextNormalizingEstimator.CaseMode.Lower;

            [Parameter(new object[] { true, false })]
            public bool keepDiacritics = false;

            [Parameter(new object[] { true, false })]
            public bool keepPunctuations = false;

            [Parameter(new object[] { true, false })]
            public bool keepNumbers = false;
        }

        public class ApplyWordEmbeddingOption : OptionBuilder<ApplyWordEmbeddingOption>
        {
            public string outputColumnName = "txt";

            public string inputColumnName = "txt";

            [Parameter(
                new object[]
                {
                    WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D,
                    WordEmbeddingEstimator.PretrainedModelKind.GloVe300D,
                    WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter100D,
                    WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding,
                })]
            public WordEmbeddingEstimator.PretrainedModelKind modelKind = WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D;
        }

        public class SdcaLogisticRegressionOption: OptionBuilder<SdcaLogisticRegressionBinaryTrainer.Options>
        {
            public string FeatureColumnName = "txt";

            public string LabelColumnName = "Sentiment";

            [Parameter(0.0001f, 10f, true, 20)]
            public float L1Regulation = 0.0001f;

            [Parameter(0.0001f, 10f, true, 20)]
            public float L2Regulation = 0.0001f;
        }
    }
}

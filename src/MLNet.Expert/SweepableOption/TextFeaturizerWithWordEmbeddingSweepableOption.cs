// <copyright file="TextFeaturizerWithWordEmbeddingSweepableOption.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Transforms.Text;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class TextFeaturizerWithWordEmbeddingSweepableOption : SweepableOption<TextFeaturizerWithWordEmbeddingSweepableOption>
    {
        public static TextFeaturizerWithWordEmbeddingSweepableOption Default = new TextFeaturizerWithWordEmbeddingSweepableOption();

        [Parameter]
        public string OutputColumnName = "txt";

        [Parameter]
        public string InputColumnName = "SentimentText";

        public TextNormalizingEstimator.CaseMode CaseMode;

        [Parameter(nameof(CaseMode))]
        public Parameter<TextNormalizingEstimator.CaseMode> CaseModeSweeper = CreateDiscreteParameter(TextNormalizingEstimator.CaseMode.Lower, TextNormalizingEstimator.CaseMode.None, TextNormalizingEstimator.CaseMode.Upper);

        public bool KeepDiacritics;

        [Parameter(nameof(KeepDiacritics))]
        public Parameter<bool> KeepDiacriticsSweeper = CreateDiscreteParameter(true, false);

        public bool KeepPunctuations;

        [Parameter(nameof(KeepPunctuations))]
        public Parameter<bool> KeepPunctuationsSweeper = CreateDiscreteParameter(true, false);

        public bool KeepNumbers;

        [Parameter(nameof(KeepNumbers))]
        public Parameter<bool> KeepNumbersSweeper = CreateDiscreteParameter(true, false);

        public WordEmbeddingEstimator.PretrainedModelKind ModelKind;

        [Parameter(nameof(ModelKind))]
        public Parameter<WordEmbeddingEstimator.PretrainedModelKind> ModelKindSweeper = CreateDiscreteParameter(
                                                                                                            WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D,
                                                                                                            WordEmbeddingEstimator.PretrainedModelKind.GloVe300D,
                                                                                                            WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter100D,
                                                                                                            WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding);
    }
}

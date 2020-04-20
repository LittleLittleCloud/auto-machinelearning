// <copyright file="ModelInput.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Data;
using System;

namespace MLNet.Examples.SentimentAnalysis
{

    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName("Sentiment")]
        public bool Sentiment;

        [LoadColumn(1)]
        [ColumnName("SentimentText")]
        public string SentimentText;
    }
}

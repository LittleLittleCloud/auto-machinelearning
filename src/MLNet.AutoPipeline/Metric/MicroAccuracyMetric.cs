// <copyright file="MicroAccuracyMetric.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline.Metric
{
    /// <summary>
    /// Calculate micro accuracy metric for multi-classification trainers.
    /// </summary>
    public class MicroAccuracyMetric : IMetric
    {
        public string Name => "MicroAccuracy";

        /// <summary>
        /// micro accuracy should be as close to 1 (max value) as possible.
        /// </summary>
        public bool IsMaximizing => true;

        /// <summary>
        /// Score model using micro accuracy. A micro accuracy is the sum of all positive predicting classes (TP and FP) divided by total predicting classes. This is more useful when dataset is imbalanced.
        /// </summary>
        /// <param name="context">ml context.</param>
        /// <param name="eval">evaluation dataview.</param>
        /// <param name="label">evaluation label column.</param>
        /// <returns>micro accuracy score.</returns>
        public double Score(MLContext context, IDataView eval, string label)
        {
            return context.MulticlassClassification.Evaluate(eval, label).MicroAccuracy;
        }
    }
}

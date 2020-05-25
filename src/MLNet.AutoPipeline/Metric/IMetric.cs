using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline.Metric
{
    /// <summary>
    /// Evaluate metric.
    /// </summary>
    public interface IMetric
    {
        /// <summary>
        /// Name for this metric.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Scoring model using <paramref name="eval"/> and <paramref name="label"/>.
        /// </summary>
        /// <param name="context">ml context.</param>
        /// <param name="eval">dataview for evaluation.</param>
        /// <param name="label">dataview column name for evaluation.</param>
        /// <returns>score.</returns>
        double Score(MLContext context, IDataView eval, string label);

        /// <summary>
        /// Indicating optimize direction.
        /// </summary>
        bool IsMaximizing { get; }
    }
}

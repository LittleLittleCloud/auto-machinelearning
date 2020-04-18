// <copyright file="RunMetric.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The metric class, used by smart sweeping algorithms.
    /// Ideally we would like to move towards a IDataView, this is
    /// just a simple view instead, and it is decoupled from RunResult so we can move
    /// in that direction in the future.
    /// </summary>
    public sealed class RunMetric
    {
        private readonly float _primaryMetric;
        private readonly float[] _metricDistribution;

        public RunMetric(float primaryMetric, IEnumerable<float> metricDistribution = null)
        {
            this._primaryMetric = primaryMetric;
            if (metricDistribution != null)
            {
                this._metricDistribution = metricDistribution.ToArray();
            }
        }

        /// <summary>
        /// The primary metric to optimize.
        /// This metric is usually an aggregate value for the run, for example, AUC, accuracy etc.
        /// By default, smart sweeping algorithms will maximize this metric.
        /// If you want to minimize, either negate this value or change the option in the arguments of the sweeper constructor.
        /// </summary>
        public float PrimaryMetric => this._primaryMetric;

        /// <summary>
        /// The (optional) distribution of the metric.
        /// This distribution can be a secondary measure of how good a run was, e.g per-fold AUC, per-fold accuracy, (sampled) per-instance log loss etc.
        /// </summary>
        public float[] GetMetricDistribution()
        {
            if (this._metricDistribution == null)
            {
                return null;
            }

            var result = new float[this._metricDistribution.Length];
            Array.Copy(this._metricDistribution, result, this._metricDistribution.Length);
            return result;
        }
    }
}

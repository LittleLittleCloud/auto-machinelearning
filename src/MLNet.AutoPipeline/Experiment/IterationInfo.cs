// <copyright file="IterationInfo.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline.Metric;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline.Experiment
{
    /// <summary>
    /// Provides information for each training round in an Experiment.
    /// </summary>
    public class IterationInfo : IComparable<IterationInfo>
    {
        /// <summary>
        /// Parameters used in the current training round.
        /// </summary>
        public ParameterSet ParameterSet { get; private set; }

        /// <summary>
        /// Training time in seconds.
        /// </summary>
        public double TrainingTime { get; private set; }

        /// <summary>
        /// Score metric name and value on validate dataset, this value is set by <see cref="Experiment.Option.ScoreMetric"/>.
        /// </summary>
        public Metric ScoreMetric { get; private set; }

        /// <summary>
        /// Evaluated metrics name and value in each sweeping, this value is set by <see cref="Experiment.Option.Metrics"/>.
        /// </summary>
        public IEnumerable<Metric> EvaluateMetrics { get; private set; }

        /// <summary>
        /// Indicate optimize direction.
        /// </summary>
        public bool IsMetricMaximizing { get; private set; }

        /// <summary>
        /// <see cref="ISweepablePipeline"/> used in the current training round.
        /// </summary>
        public ISweepablePipeline SweepablePipeline { get; private set; }

        public IterationInfo(ISweepablePipeline sweepablePipeline, ParameterSet parameters, double time, Metric score, IEnumerable<Metric> metrics, bool isMaximizing)
        {
            this.SweepablePipeline = sweepablePipeline;
            this.ParameterSet = parameters;
            this.TrainingTime = time;
            this.ScoreMetric = score;
            this.EvaluateMetrics = metrics;
            this.IsMetricMaximizing = isMaximizing;
        }

        /// <summary>
        /// Restore untrained pipeline from <see cref="ParameterSet"/>.
        /// </summary>
        /// <returns></returns>
        public EstimatorChain<ITransformer> BuildPipeline()
        {
            return (this.SweepablePipeline as SweepablePipeline)?.BuildFromParameterSet(this.ParameterSet);
        }

        public int CompareTo(IterationInfo obj)
        {
            if (obj is null || obj?.ScoreMetric.Name != this.ScoreMetric.Name)
            {
                return 1;
            }

            if (this.IsMetricMaximizing)
            {
                return this.ScoreMetric.Score.CompareTo(obj.ScoreMetric.Score);
            }
            else
            {
                return obj.ScoreMetric.Score.CompareTo(this.ScoreMetric.Score);
            }
        }

        public static bool operator > (IterationInfo info1, IterationInfo info2)
        {
            return info1.CompareTo(info2) == 1;
        }

        public static bool operator < (IterationInfo info1, IterationInfo info2)
        {
            return info1.CompareTo(info2) == -1;
        }

        public static bool operator >=(IterationInfo info1, IterationInfo info2)
        {
            return info1.CompareTo(info2) >= 0;
        }

        public static bool operator <=(IterationInfo info1, IterationInfo info2)
        {
            return info1.CompareTo(info2) <= 0;
        }

        public class Metric
        {
            public Metric(string name, double score)
            {
                this.Name = name;
                this.Score = score;
            }

            public string Name { get; set; }

            public double Score { get; set; }
        }
    }

}

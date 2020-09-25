// <copyright file="IterationInfo.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Provides information for each training round in an Experiment.
    /// </summary>
    public class IterationInfo : IComparable<IterationInfo>
    {
        /// <summary>
        /// Parameters used in the current training round. This value is generated through <see cref="ISweeper.ProposeSweeps(ISweepable, int, IEnumerable{IRunResult})"/> from <see cref="Experiment.Option.ParameterSweeper"/>.
        /// </summary>
        public IDictionary<string, string> Parameters { get; private set; }

        /// <summary>
        /// Training time in seconds.
        /// </summary>
        public double TrainingTime { get; private set; }

        /// <summary>
        /// Score metric name and value on validate dataset, this value is set by <see cref="Experiment.Option.EvaluateFunction"/>.
        /// </summary>
        public double EvaluateScore { get; private set; }

        public double[] Metrics { get; private set; }

        public ITransformer Model { get; }

        /// <summary>
        /// Indicate optimize direction.
        /// </summary>
        public bool IsMetricMaximizing { get; private set; }

        /// <summary>
        /// <see cref="SingleSweepablePipeline"/> used in the current training round.
        /// </summary>
        public SingleEstimatorSweepablePipeline SingleSweepablePipeline { get; private set; }

        internal IterationInfo(SingleEstimatorSweepablePipeline singleweepablePipeline, IDictionary<string, string> parameters, double time, double evaluateScore, bool isMaximizing, double[] metrics = null, ITransformer model = null)
        {
            this.SingleSweepablePipeline = singleweepablePipeline;
            this.Parameters = parameters;
            this.TrainingTime = time;
            this.EvaluateScore = evaluateScore;
            this.IsMetricMaximizing = isMaximizing;
            this.Metrics = metrics ?? new double[0];
            this.Model = model;
        }

        /// <summary>
        /// Restore MLNet estimator chain from <see cref="Parameters"/>.
        /// </summary>
        /// <returns><see cref="EstimatorChain{TLastTransformer}"/>.</returns>
        public EstimatorChain<ITransformer> RestoreEstimatorChain()
        {
            return this.SingleSweepablePipeline.BuildFromParameters(this.Parameters);
        }

        public int CompareTo(IterationInfo obj)
        {
            if (obj is null)
            {
                return 1;
            }

            if (this.IsMetricMaximizing)
            {
                return this.EvaluateScore.CompareTo(obj.EvaluateScore);
            }
            else
            {
                return obj.EvaluateScore.CompareTo(this.EvaluateScore);
            }
        }

        public static bool operator >(IterationInfo info1, IterationInfo info2)
        {
            return info1.CompareTo(info2) == 1;
        }

        public static bool operator <(IterationInfo info1, IterationInfo info2)
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

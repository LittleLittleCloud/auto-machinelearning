// <copyright file="IterationInfo.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
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
        public ParameterSet Parameters { get; private set; }

        /// <summary>
        /// Training time in seconds.
        /// </summary>
        public float TrainingTime { get; private set; }

        /// <summary>
        /// Training score.
        /// </summary>
        public double Score { get; private set; }

        /// <summary>
        /// Indicate optimize direction.
        /// </summary>
        public bool IsMaximizing { get; private set; }

        /// <summary>
        /// <see cref="ISweepablePipeline"/> used in the current training round.
        /// </summary>
        public ISweepablePipeline SweepablePipeline { get; private set; }

        public IterationInfo(ISweepablePipeline sweepablePipeline, ParameterSet parameters, float time, double score, bool isMaximizing)
        {
            this.SweepablePipeline = sweepablePipeline;
            this.Parameters = parameters;
            this.TrainingTime = time;
            this.Score = score;
            this.IsMaximizing = isMaximizing;
        }

        /// <summary>
        /// Restore untrained pipeline from <see cref="Parameters"/>.
        /// </summary>
        /// <returns></returns>
        public EstimatorChain<ITransformer> BuildPipeline()
        {
            return (this.SweepablePipeline as SweepablePipeline)?.BuildFromParameterSet(this.Parameters);
        }

        public int CompareTo(IterationInfo obj)
        {
            if (obj is null)
            {
                return 1;
            }

            if (this.IsMaximizing)
            {
                return this.Score.CompareTo(obj.Score);
            }
            else
            {
                return obj.Score.CompareTo(this.Score);
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
    }
}

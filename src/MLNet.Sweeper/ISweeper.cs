// <copyright file="ISweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Internal.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    public interface ISweeper
    {
        /// <summary>
        /// Returns between 0 and maxSweeps configurations to run.
        /// It expects a list of previous runs such that it can generate configurations that were not already tried.
        /// The list of runs can be null if there were no previous runs.
        /// Some smart sweepers can take advantage of the metric(s) that the caller computes for previous runs.
        /// </summary>
        /// <returns>ParameterSet.</returns>
        ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null);
    }

    public interface ISweepResultEvaluator<in TResults>
    {
        /// <summary>
        /// Return an IRunResult based on the results given as a TResults object.
        /// </summary>
        IRunResult GetRunResult(ParameterSet parameters, TResults results);
    }

    /// <summary>
    /// The result of a run.
    /// Contains the parameter set used, useful for the sweeper to not generate the same configuration multiple times.
    /// Also contains the result of a run and the metric value that is used by smart sweepers to generate new configurations
    /// that try to maximize this metric.
    /// </summary>
    public interface IRunResult : IComparable<IRunResult>
    {
        ParameterSet ParameterSet { get; }

        IComparable MetricValue { get; }

        bool IsMetricMaximizing { get; }
    }

    public interface IRunResult<T> : IRunResult
        where T : IComparable<T>
    {
        new T MetricValue { get; }
    }
}

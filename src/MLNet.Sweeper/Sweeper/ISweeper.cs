// <copyright file="ISweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    public interface ISweeper : ICloneable
    {
        /// <summary>
        /// Returns between 0 and maxSweeps configurations to run.
        /// It expects a list of previous runs such that it can generate configurations that were not already tried.
        /// The list of runs can be null if there were no previous runs.
        /// Some smart sweepers can take advantage of the metric(s) that the caller computes for previous runs.
        /// </summary>
        /// <returns><see cref="IDictionary{TKey, TValue}"/> where key is parameter name and value is parameter value (in string format).</returns>
        IEnumerable<IDictionary<string, string>> ProposeSweeps(ISweepable sweepingSpace, int maxSweeps = 100, IEnumerable<IRunResult> previousRuns = null);

        /// <summary>
        /// Add run history to sweeper. The run history can be used to avoid duplicate sweeping or train smart sweepers.
        /// </summary>
        /// <param name="input">sweeping result.</param>
        void AddRunHistory(IRunResult input);
    }

    /// <summary>
    /// The result of a run.
    /// Contains the parameter set used, useful for the sweeper to not generate the same configuration multiple times.
    /// Also contains the result of a run and the metric value that is used by smart sweepers to generate new configurations
    /// that try to maximize this metric.
    /// </summary>
    public interface IRunResult : IComparable<IRunResult>
    {
        IDictionary<string, string> ParameterSet { get; }

        IComparable MetricValue { get; }

        bool IsMetricMaximizing { get; }
    }

    public interface IRunResult<T> : IRunResult
        where T : IComparable<T>
    {
        new T MetricValue { get; }
    }

    public interface ISweepable
    {
        IEnumerable<IValueGenerator> SweepableValueGenerators { get; }
    }

    internal interface ISweepable<out T> : ISweepable
    {
        T BuildFromParameters(IDictionary<string, string> parameters);
    }
}
